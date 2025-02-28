MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
data_dir="/home/wyh/HDD2/wyh/UltrasoundDataWithMRI/AllSA1ImageResult512x512"
RESUMECHECKPOINT="/home/wyh/Project/FusionProject/MMIF-DDFM/DiffusionModel/PretrainedWeight/256x256_diffusion.pt"
log_dir="/home/wyh/Project/FusionProject/MMIF-DDFM/DiffusionModel/ExpResult/LOG"
NUM_GPUS=2
# python finetune.py --data_dir $data_dir $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --resume_checkpoint $RESUMECHECKPOINT --log_dir $log_dir
# mpiexec -n $NUM_GPUS python finetune.py --data_dir $data_dir $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --resume_checkpoint $RESUMECHECKPOINT --log_dir $log_dir
torchrun --nproc_per_node=$NUM_GPUS finetune.py \
    --data_dir $data_dir \
    $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
    --resume_checkpoint $RESUMECHECKPOINT \
    --log_dir $log_dir \
    --use_torch_distributed