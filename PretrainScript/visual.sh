# 显示更多图像，并保存到文件
npzpath="/home/wyh/wyhHDD/DiffusionProject/PretrainedWeights/sampling/samples_25x256x256x3.npz"
save_path=$(dirname $npzpath)
save_path="${save_path}/output.png"
python visualSampledResult.py --npz_path $npzpath --num_images 25 --grid_size 5,5 --save_path $save_path