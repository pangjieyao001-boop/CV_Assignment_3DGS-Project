"""
可视化工具
保存图像、创建视频、绘制图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from pathlib import Path
from typing import List, Dict


def save_image(image: np.ndarray or torch.Tensor, path: str, normalize=False):
    """
    保存图像
    
    Args:
        image: 图像数据 [H, W, 3] 或 [3, H, W]
        path: 保存路径
        normalize: 是否归一化到[0, 1]
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # 确保通道在最后
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 确保范围在[0, 1]或[0, 255]
    if normalize:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # 保存
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).save(path)


def create_video(image_paths: List[str], output_path: str, fps=30):
    """
    从图像序列创建视频
    
    Args:
        image_paths: 图像路径列表
        output_path: 输出视频路径
        fps: 帧率
    """
    try:
        import cv2
        
        # 读取第一张图像获取尺寸
        first_frame = cv2.imread(image_paths[0])
        height, width, _ = first_frame.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")
    except ImportError:
        print("OpenCV not available, skipping video creation")


def plot_training_loss(losses: List[float], output_path: str):
    """
    绘制训练损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Loss plot saved to {output_path}")


def plot_metrics_comparison(results: Dict[str, Dict], output_path: str):
    """
    绘制不同方法的指标对比
    
    Args:
        results: 字典，格式为 {method_name: {'psnr': x, 'ssim': y, 'lpips': z, 'size_mb': s}}
        output_path: 输出路径
    """
    methods = list(results.keys())
    metrics = ['psnr', 'ssim', 'lpips', 'size_mb']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[m].get(metric, 0) for m in methods]
        
        bars = ax.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(methods)])
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Metrics comparison saved to {output_path}")


def plot_rate_distortion(results: Dict[str, Dict], output_path: str):
    """
    绘制率-失真曲线 (Rate-Distortion Curve)
    
    Args:
        results: 字典，格式为 {method_name: {'psnr': x, 'size_mb': y}}
        output_path: 输出路径
    """
    plt.figure(figsize=(10, 6))
    
    for method_name, data in results.items():
        psnr = data.get('psnr', 0)
        size_mb = data.get('size_mb', 0)
        plt.plot(size_mb, psnr, 'o-', label=method_name, markersize=8)
    
    plt.xlabel('Model Size (MB)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Rate-distortion curve saved to {output_path}")


def plot_compression_ratio_comparison(results: Dict[str, Dict], output_path: str):
    """
    绘制压缩比对比
    """
    methods = list(results.keys())
    compression_ratios = [results[m].get('compression_ratio', 1.0) for m in methods]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, compression_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio Comparison (Higher is Better)')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, val in zip(bars, compression_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Compression ratio comparison saved to {output_path}")


def visualize_gaussian_distribution(model, output_path: str):
    """
    可视化Gaussian的空间分布
    """
    xyz = model.get_xyz.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Gaussian Distribution')
    
    # XY平面
    ax2 = fig.add_subplot(132)
    ax2.scatter(xyz[:, 0], xyz[:, 1], s=1, alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane View')
    ax2.axis('equal')
    
    # XZ平面
    ax3 = fig.add_subplot(133)
    ax3.scatter(xyz[:, 0], xyz[:, 2], s=1, alpha=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane View')
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gaussian distribution visualization saved to {output_path}")


def create_comparison_grid(images_dict: Dict[str, np.ndarray], output_path: str, titles=None):
    """
    创建对比图像网格
    
    Args:
        images_dict: 字典，格式为 {name: image}
        output_path: 输出路径
        titles: 标题列表
    """
    n_images = len(images_dict)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (name, img) in enumerate(images_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        
        # 确保图像格式正确
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        ax.imshow(img)
        ax.set_title(name if titles is None else titles[idx])
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to {output_path}")


def plot_ablation_study(results: Dict[str, Dict], output_path: str):
    """
    绘制消融实验结果
    
    Args:
        results: 字典，格式为 {
            'baseline': {'psnr': x, 'ssim': y},
            'w/o_pruning': {...},
            ...
        }
    """
    methods = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['psnr', 'ssim', 'lpips']
    titles = ['PSNR (dB)', 'SSIM', 'LPIPS']
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [results[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, values, color='steelblue')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Ablation study plot saved to {output_path}")
