SAMPLE_FLAGS="--batch_size 4 --num_samples 10 --timestep_respacing 250"
# SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing ddim25 --use_ddim True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
log_dir="/home/wyh/wyhHDD/DiffusionProject/Result/FinetuneV2/samplling"
modelpath="/home/wyh/wyhHDD/DiffusionProject/Result/FinetuneV2/model010000.pt"
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 samplingfinetune.py $MODEL_FLAGS --model_path $modelpath $SAMPLE_FLAGS --log_dir $log_dir