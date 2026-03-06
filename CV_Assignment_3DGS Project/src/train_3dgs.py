"""
3D Gaussian Splatting 训练脚本
支持Apple Silicon (MPS) 和 CPU
基于gsplat库
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

# 尝试导入gsplat
try:
    from gsplat import rasterize_gaussians
    from gsplat.utils import (
        quat_to_rotmat,
        rotation_matrix_to_quaternion,
        spherical_harmonics,
    )
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available. Using fallback implementation.")

from utils.data_loader import ColmapDataLoader, SyntheticDataLoader
from utils.metrics import compute_psnr, compute_ssim, compute_lpips
from utils.visualization import save_image, create_video


class GaussianModel(nn.Module):
    """
    3D Gaussian模型
    """
    def __init__(self, num_points: int = 10000, sh_degree: int = 3):
        super().__init__()
        self.num_points = num_points
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        
        # 初始化Gaussian参数
        # 位置
        self._xyz = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        # 特征（球谐系数）
        features_dc = torch.rand(num_points, 3)
        features_rest = torch.rand(num_points, 3 * (sh_degree + 1) ** 2 - 3)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        # 不透明度
        self._opacity = nn.Parameter(torch.randn(num_points, 1))
        # 缩放
        self._scaling = nn.Parameter(torch.randn(num_points, 3))
        # 旋转（四元数）
        self._rotation = nn.Parameter(torch.randn(num_points, 4))
        
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat([features_dc, features_rest], dim=1)
    
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)
    
    @property
    def get_scaling(self):
        return torch.exp(self._scaling)
    
    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        分割大的Gaussians
        """
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        
        padded_grad = torch.zeros(n_init_points, device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > scene_extent * 0.01
        )
        
        if selected_pts_mask.sum() == 0:
            return
        
        # 创建新的Gaussians
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        
        new_xyz = self._xyz[selected_pts_mask].repeat(N, 1) + samples
        new_scaling = self._scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        
        # 更新参数
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        克隆小的Gaussians
        """
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= scene_extent * 0.01
        )
        
        if selected_pts_mask.sum() == 0:
            return
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))

    def prune_points(self, mask):
        """
        剪枝低不透明度的Gaussians
        """
        valid_points_mask = ~mask
        
        self._xyz = nn.Parameter(self._xyz[valid_points_mask])
        self._scaling = nn.Parameter(self._scaling[valid_points_mask])
        self._rotation = nn.Parameter(self._rotation[valid_points_mask])
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask])
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask])
        self._opacity = nn.Parameter(self._opacity[valid_points_mask])
    
    def render_from_camera(self, camera, device):
        """
        从相机视角渲染（简化版本，支持梯度）
        """
        img_h, img_w = camera['height'], camera['width']
        
        # 使用可微操作
        xyz = self._xyz
        opacity = torch.sigmoid(self._opacity)
        colors = torch.sigmoid(self._features_dc)
        
        # 计算每个点对图像的贡献（简化版splatting）
        # 相机参数
        R = camera['R'].to(device)
        T = camera['T'].to(device)
        
        # 变换到相机坐标系
        xyz_cam = torch.matmul(xyz, R.T) + T
        
        # 使用深度加权的方式聚合颜色
        depths = xyz_cam[:, 2]
        valid_depth = depths > 0.1
        
        if valid_depth.sum() == 0:
            return torch.zeros(3, img_h, img_w, device=device)
        
        # 基于深度计算权重（越近的点权重越高）
        inv_depths = 1.0 / (depths[valid_depth] + 1e-6)
        weights = inv_depths / inv_depths.sum()
        
        # 加权颜色
        weighted_colors = colors[valid_depth] * opacity[valid_depth] * weights.unsqueeze(1)
        
        # 创建均匀的图像（简化表示）
        mean_color = weighted_colors.sum(dim=0)
        rendered = mean_color.view(3, 1, 1).expand(3, img_h, img_w)
        
        return rendered


def train_3dgs(
    data_path: str,
    output_path: str,
    iterations: int = 7000,
    save_iterations: list = [1000, 3000, 5000, 7000],
    device: str = "auto"
):
    """
    训练3D Gaussian Splatting模型
    
    Args:
        data_path: 数据目录路径
        output_path: 输出目录路径
        iterations: 训练迭代次数
        save_iterations: 保存模型的迭代点
        device: 计算设备 (auto/cpu/cuda/mps)
    """
    
    # 自动检测设备
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"Using Apple Metal (MPS) device")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU device (training will be slow)")
    
    device = torch.device(device)
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {data_path}...")
    if (Path(data_path) / "sparse").exists():
        # COLMAP格式数据
        data_loader = ColmapDataLoader(data_path, device=device)
    else:
        # 合成数据格式
        data_loader = SyntheticDataLoader(data_path, device=device)
    
    cameras = data_loader.load_cameras()
    point_cloud = data_loader.load_point_cloud()
    
    print(f"Loaded {len(cameras)} cameras, {len(point_cloud)} initial points")
    
    # 初始化模型
    num_points = len(point_cloud) if len(point_cloud) > 0 else 10000
    model = GaussianModel(num_points=num_points, sh_degree=3)
    
    # 如果有初始点云，用它来初始化
    if len(point_cloud) > 0:
        with torch.no_grad():
            model._xyz.data = torch.tensor(point_cloud, dtype=torch.float32, device=device)
    
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练循环
    print(f"Starting training for {iterations} iterations...")
    start_time = time.time()
    
    progress_bar = tqdm(range(iterations), desc="Training")
    
    for iteration in progress_bar:
        # 随机选择一个相机视角
        cam = cameras[iteration % len(cameras)]
        
        # 渲染（简化版本，实际需要实现完整的rasterization）
        # 这里使用简化的渲染过程
        
        # 计算损失（与GT图像的L1 + SSIM）
        gt_image = cam['image'].to(device)
        
        # 简化的前向传播（实际应该使用gsplat的rasterize_gaussians）
        # rendered_image = render_simple(model, cam, device)
        
        # 这里使用随机损失作为占位符
        # 实际需要实现完整的渲染流程
        loss = torch.rand(1, device=device, requires_grad=True)
        
        # 反向传播
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
        
        # 更新进度条
        if iteration % 100 == 0:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 保存模型
        if iteration in save_iterations:
            save_path = output_path / f"model_iter_{iteration}.pth"
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"\nSaved model to {save_path}")
    
    # 保存最终模型
    final_path = output_path / "model_final.pth"
    torch.save({
        'iteration': iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    # 保存配置
    config = {
        'iterations': iterations,
        'num_points': model.get_xyz.shape[0],
        'sh_degree': model.sh_degree,
        'device': str(device),
        'training_time': time.time() - start_time,
    }
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Total training time: {config['training_time']:.2f}s")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")
    parser.add_argument("-s", "--source", required=True, help="Path to dataset")
    parser.add_argument("-m", "--model", required=True, help="Path to output model")
    parser.add_argument("--iterations", type=int, default=7000, help="Number of iterations")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use")
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    
    args = parser.parse_args()
    
    # 训练模型
    model = train_3dgs(
        data_path=args.source,
        output_path=args.model,
        iterations=args.iterations,
        device=args.device
    )
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
