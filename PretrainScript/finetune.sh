MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_kl True"
EMA_RATE="--ema_rate 0.9999,0.999"
TRAIN_FLAGS="--lr 0.5e-4 --batch_size 1 --weight_decay 0.001 --lr_anneal_steps 40000"
data_dir="/home/wyh/wyhHDD/DiffusionProject/Dataset/AllSA1ImageResult256x256"
RESUMECHECKPOINT="/home/wyh/wyhHDD/DiffusionProject/PretrainedWeights/256x256_diffusion.pt"
log_dir="/home/wyh/wyhHDD/DiffusionProject/Result/FinetuneProperDiffSetNoCond_256x256"
NUM_GPUS=4
# python finetune.py --data_dir $data_dir $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --resume_checkpoint $RESUMECHECKPOINT --log_dir $log_dir
# mpiexec -n $NUM_GPUS python finetune.py --data_dir $data_dir $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --resume_checkpoint $RESUMECHECKPOINT --log_dir $log_dir
CUDA_VISIBLE_DEVICES=1,3,4,5 torchrun --nproc_per_node=$NUM_GPUS finetune.py \
    --data_dir $data_dir \
    $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
    --resume_checkpoint $RESUMECHECKPOINT \
    --log_dir $log_dir --in_channels 1 --out_channels 1 --finetuneFlag True \
    --project_name FineTune --exp_name FinetuneProperDiffSetNoCond_256x256 $EMA_RATE