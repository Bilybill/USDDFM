model_config="/home/wyh/Project/FusionProject/MMIF-DDFM/configs/model_config_usdffm.yaml"
diffusion_config="/home/wyh/Project/FusionProject/MMIF-DDFM/configs/diffusion_config_usdffm.yaml"
gpu=0
save_dir="/home/wyh/HDD2/wyh/UltrasoundDataWithMRI/TestregisInput/save_dir"
python sample.py --model_config $model_config --diffusion_config $diffusion_config --gpu $gpu --save_dir $save_dir