import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        project_wandb=None,
        exp_name_wandb=None,
        finetune_flag=False
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self.finetune_flag = finetune_flag

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        if project_wandb is not None and exp_name_wandb is not None and dist.get_rank() == 0:
            wandb.init(
                project=project_wandb, 
                name=exp_name_wandb,
                dir=get_blob_logdir()
            )
            wandb.config.update({
                "batch_size": self.batch_size,
                "microbatch": self.microbatch,
                "lr": self.lr,
                "ema_rate": self.ema_rate,
                "log_interval": self.log_interval,
                "save_interval": self.save_interval,
                "resume_checkpoint": self.resume_checkpoint,
                "use_fp16": self.use_fp16,
                "fp16_scale_growth": self.fp16_scale_growth,
                "weight_decay": self.weight_decay,
                "lr_anneal_steps": self.lr_anneal_steps
            })
                

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
                
                if not self.finetune_flag:
                    self.model.load_state_dict(state_dict)
                else:
                    # 预处理状态字典，处理形状不匹配的情况
                    filtered_state_dict = {}
                    shape_mismatch_keys = []
                    
                    # 检查模型参数与加载的状态字典
                    model_state = self.model.state_dict()
                    for k, v in state_dict.items():
                        if k in model_state:
                            if v.shape != model_state[k].shape:
                                shape_mismatch_keys.append(k)
                                logger.log(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_state[k].shape}")
                            else:
                                filtered_state_dict[k] = v
                        else:
                            # 键不在模型中，会成为unexpected_keys
                            filtered_state_dict[k] = v
                    
                    # 在微调模式下加载模型，strict=False，并记录警告信息
                    load_result = self.model.load_state_dict(filtered_state_dict, strict=False)
                    
                    # 记录缺失的键(未加载的层)
                    if load_result.missing_keys:
                        logger.log(f"Warning: {len(load_result.missing_keys)} missing keys in state_dict:")
                        for key in load_result.missing_keys:
                            logger.log(f"  - Missing key (will be randomly initialized): {key}")
                    
                    # 记录多余的键(预训练模型中有但当前模型中没有的层)
                    if load_result.unexpected_keys:
                        logger.log(f"Warning: {len(load_result.unexpected_keys)} unexpected keys in state_dict:")
                        for key in load_result.unexpected_keys:
                            logger.log(f"  - Unexpected key (will be ignored): {key}")
                    
                    # 记录形状不匹配的键
                    if shape_mismatch_keys:
                        logger.log(f"Warning: {len(shape_mismatch_keys)} shape mismatched keys (will be randomly initialized):")
                        for key in shape_mismatch_keys:
                            logger.log(f"  - Shape mismatch key: {key}")
                    
                    # 同时记录到wandb日志中
                    if hasattr(self, 'wandb') and wandb.run is not None:
                        wandb.log({
                            "finetune/missing_keys_count": len(load_result.missing_keys),
                            "finetune/unexpected_keys_count": len(load_result.unexpected_keys),
                            "finetune/shape_mismatch_count": len(shape_mismatch_keys),
                        })
                        # 如果键太多，只记录前10个示例
                        missing_examples = load_result.missing_keys[:10] if load_result.missing_keys else []
                        unexpected_examples = load_result.unexpected_keys[:10] if load_result.unexpected_keys else []
                        shape_mismatch_examples = shape_mismatch_keys[:10] if shape_mismatch_keys else []
                        
                        wandb.log({
                            "finetune/missing_keys_examples": ", ".join(missing_examples),
                            "finetune/unexpected_keys_examples": ", ".join(unexpected_examples),
                            "finetune/shape_mismatch_examples": ", ".join(shape_mismatch_examples),
                        })

            dist_util.sync_params(self.model.parameters())
            logger.log("model loaded and synced")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        wandb.finish()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
            
        # Log the LR
        if dist.get_rank() == 0:
            logger.logkv("learning_rate", lr)
            wandb.log({"learning_rate": lr})

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        # 仅主进程向 wandb 记录
        if dist.get_rank() == 0:
            wandb.log({
                "step": self.step + self.resume_step,
                "samples": (self.step + self.resume_step + 1) * self.global_batch,
            })

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    # Compute mean loss across all devices
    for key, values in losses.items():
        # Reduce across all processes for accurate loss
        if dist.is_initialized():
            values_tensor = values.clone().detach()
            dist.all_reduce(values_tensor)
            values_tensor /= dist.get_world_size()
            mean_value = values_tensor.mean().item()
        else:
            mean_value = values.mean().item()
        
        logger.logkv_mean(key, mean_value)
        
        # Only log to wandb from main process
        if dist.get_rank() == 0:
            wandb.log({key: mean_value})
            
        # Process quartiles
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            if dist.get_rank() == 0:
                wandb.log({f"{key}_q{quartile}": sub_loss})
