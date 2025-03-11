num_samples=25
in_channels=1
out_channels=1
imgsize=256
# SAMPLE_FLAGS="--batch_size 4 --num_samples $num_samples --timestep_respacing 250"
SAMPLE_FLAGS="--batch_size 4 --num_samples $num_samples --timestep_respacing ddim25 --use_ddim True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
modelpath="/home/wyh/wyhHDD/DiffusionProject/Result/FinetuneProperDiffSetNoCond_256x256/model040000.pt"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
# ---------------------------------------------------------------------------------------------------------------
# parse modelpath to get log_dir
log_dir=$(dirname $modelpath)
log_dir="${log_dir}/sampling"
echo "log_dir: to $log_dir"
# 使用随机端口避免冲突
RANDOM_PORT=$((29500 + RANDOM % 10000))
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port $RANDOM_PORT samplingfinetune.py $MODEL_FLAGS --model_path $modelpath $SAMPLE_FLAGS --log_dir $log_dir --in_channels $in_channels --out_channels $out_channels $DIFFUSION_FLAGS
npzpath="${log_dir}/samples_${num_samples}x${imgsize}x${imgsize}x${out_channels}.npz"
echo "load npz file from $npzpath"
save_path=$(dirname $npzpath)
loadmodelname=$(basename $modelpath)
#remove the .pt suffix
loadmodelname="${loadmodelname%.*}"
save_path="${save_path}/${loadmodelname}.png"
python visualSampledResult.py --npz_path $npzpath --num_images 25 --grid_size 5,5 --save_path $save_path