#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化由扩散模型生成的样本
用法: python visualize_samples.py --npz_path path/to/samples.npz --num_images 16 --grid_size 4,4 --save_path output.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


def main():
    parser = argparse.ArgumentParser(description='扩散模型样本可视化工具')
    parser.add_argument('--npz_path', type=str, required=True, help='NPZ文件路径')
    parser.add_argument('--num_images', type=int, default=16, help='要可视化的图片数量')
    parser.add_argument('--grid_size', type=str, default=None, help='网格大小，格式如 "4,4"')
    parser.add_argument('--random_seed', type=int, default=None, help='随机种子，用于随机选择图像')
    parser.add_argument('--save_path', type=str, default=None, help='保存可视化图像的路径')
    parser.add_argument('--show_labels', action='store_true', help='如果有标签，显示标签')
    args = parser.parse_args()

    # 加载NPZ文件
    npz_data = np.load(args.npz_path)
    
    # 确定图像数组和标签数组
    if len(npz_data.files) > 1:
        # 假设条件生成，数组有两个：图像和标签
        images = npz_data[npz_data.files[0]]
        labels = npz_data[npz_data.files[1]]
        has_labels = True
    else:
        # 非条件生成，只有图像数组
        images = npz_data[npz_data.files[0]]
        labels = None
        has_labels = False
    
    total_images = images.shape[0]
    print(f"NPZ文件包含 {total_images} 张图像，形状为 {images.shape}")
    
    # 设置随机种子
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    # 随机选择图像索引
    if args.num_images < total_images:
        indices = np.random.choice(total_images, size=args.num_images, replace=False)
        images = images[indices]
        if has_labels and labels is not None:
            labels = labels[indices]
    else:
        args.num_images = min(args.num_images, total_images)
        images = images[:args.num_images]
        if has_labels and labels is not None:
            labels = labels[:args.num_images]
    
    # 确定网格大小
    if args.grid_size:
        rows, cols = map(int, args.grid_size.split(','))
        if rows * cols < args.num_images:
            print(f"警告: 网格大小 {rows}x{cols} 小于图像数量 {args.num_images}，将调整网格大小")
            rows = cols = int(math.ceil(math.sqrt(args.num_images)))
    else:
        rows = cols = int(math.ceil(math.sqrt(args.num_images)))
    
    # 创建网格图
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    
    # 处理单图情况
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    # 填充图像
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < args.num_images:
                axes[i, j].imshow(images[idx], cmap = 'gray')
                if has_labels and labels is not None and args.show_labels:
                    axes[i, j].set_title(f"Label: {labels[idx]}")
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if args.save_path:
        plt.savefig(args.save_path, bbox_inches='tight')
        print(f"图像已保存至: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()