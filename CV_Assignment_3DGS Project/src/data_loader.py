"""
数据加载器
支持COLMAP格式和合成数据格式
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional


class Camera:
    """相机类"""
    def __init__(self, uid, R, T, FoVx, FoVy, image, gt_alpha_mask=None,
                 image_name=None, uid_id=None, data_device="cpu"):
        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image = image
        self.gt_alpha_mask = gt_alpha_mask
        self.image_name = image_name
        self.uid_id = uid_id
        
        try:
            self.data_device = torch.device(data_device)
        except Exception:
            self.data_device = torch.device("cpu")
        
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width),
                                              device=self.data_device)


class ColmapDataLoader:
    """
    COLMAP格式数据加载器
    """
    def __init__(self, data_path: str, device="cpu", resize=None):
        self.data_path = Path(data_path)
        self.device = device
        self.resize = resize
        
    def load_cameras(self) -> List[Dict]:
        """
        加载相机参数
        """
        cameras = []
        
        # 读取COLMAP的cameras.txt和images.txt
        sparse_path = self.data_path / "sparse" / "0"
        
        if not sparse_path.exists():
            print(f"Warning: Sparse directory not found at {sparse_path}")
            return cameras
        
        # 解析cameras.txt
        cameras_txt = sparse_path / "cameras.txt"
        images_txt = sparse_path / "images.txt"
        
        if not cameras_txt.exists() or not images_txt.exists():
            print(f"Warning: COLMAP files not found")
            return cameras
        
        # 读取相机内参
        camera_params = {}
        with open(cameras_txt, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    cam_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(x) for x in parts[4:]]
                    camera_params[cam_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'params': params
                    }
        
        # 读取图像位姿
        images_dir = self.data_path / "images"
        
        with open(images_txt, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            
            parts = line.split()
            if len(parts) >= 9:
                img_id = int(parts[0])
                qw, qx, qy, qz = [float(x) for x in parts[1:5]]
                tx, ty, tz = [float(x) for x in parts[5:8]]
                cam_id = int(parts[8])
                img_name = parts[9] if len(parts) > 9 else f"image_{img_id:04d}.png"
                
                # 构建旋转矩阵（四元数转旋转矩阵）
                R = self._quat_to_rotmat([qw, qx, qy, qz])
                T = np.array([tx, ty, tz])
                
                # 加载图像
                img_path = images_dir / img_name
                if img_path.exists():
                    img = Image.open(img_path).convert('RGB')
                    if self.resize:
                        img = img.resize(self.resize)
                    img_array = np.array(img) / 255.0
                    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
                    
                    # 获取相机参数
                    cam_param = camera_params.get(cam_id, {})
                    if cam_param:
                        fx = cam_param['params'][0] if cam_param['params'] else 1000.0
                        fy = fx
                        if len(cam_param['params']) > 1:
                            fy = cam_param['params'][1]
                        
                        width = cam_param['width']
                        height = cam_param['height']
                        
                        FoVx = 2 * np.arctan(width / (2 * fx))
                        FoVy = 2 * np.arctan(height / (2 * fy))
                        
                        cameras.append({
                            'uid': img_id,
                            'R': torch.from_numpy(R).float(),
                            'T': torch.from_numpy(T).float(),
                            'FoVx': FoVx,
                            'FoVy': FoVy,
                            'image': img_tensor,
                            'image_name': img_name,
                            'width': width,
                            'height': height
                        })
            
            i += 2  # 跳过POINTS2D行
        
        return cameras
    
    def load_point_cloud(self) -> np.ndarray:
        """
        加载点云（从points3D.txt）
        """
        sparse_path = self.data_path / "sparse" / "0"
        points3d_txt = sparse_path / "points3D.txt"
        
        if not points3d_txt.exists():
            print(f"Warning: points3D.txt not found")
            return np.array([])
        
        points = []
        with open(points3d_txt, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, z = [float(parts[i]) for i in range(1, 4)]
                    r, g, b = [int(parts[i]) for i in range(4, 7)]
                    points.append([x, y, z, r, g, b])
        
        return np.array(points) if points else np.array([])
    
    def _quat_to_rotmat(self, quat):
        """四元数转旋转矩阵"""
        qw, qx, qy, qz = quat
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R


class SyntheticDataLoader:
    """
    合成数据加载器（NeRF格式）
    """
    def __init__(self, data_path: str, device="cpu", resize=None):
        self.data_path = Path(data_path)
        self.device = device
        self.resize = resize
        
    def load_cameras(self, split='train') -> List[Dict]:
        """
        加载相机参数
        """
        cameras = []
        
        # 读取transforms.json
        transforms_path = self.data_path / f"transforms_{split}.json"
        
        if not transforms_path.exists():
            print(f"Warning: {transforms_path} not found")
            return cameras
        
        with open(transforms_path, 'r') as f:
            meta = json.load(f)
        
        camera_angle_x = meta.get('camera_angle_x', 0.6911112070083618)
        
        for i, frame in enumerate(meta.get('frames', [])):
            file_path = frame['file_path']
            
            # 处理 NerfBaselines 格式
            # 移除开头的 ./
            if file_path.startswith('./'):
                file_path = file_path[2:]
            
            img_name = file_path.split('/')[-1]
            
            # 添加 .png 后缀
            if not img_name.endswith('.png'):
                img_name += '.png'
                file_path = file_path + '.png'
            
            # 跳过 depth 图像
            if '_depth_' in img_name or img_name.endswith('_depth.png'):
                continue
            
            # 使用完整的文件路径
            img_path = self.data_path / file_path
            
            if not img_path.exists():
                # 尝试在 train/test 目录中查找
                for subdir in ['train', 'test', 'val']:
                    alt_path = self.data_path / subdir / img_name
                    if alt_path.exists():
                        img_path = alt_path
                        break
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            if self.resize:
                img = img.resize(self.resize)
            img_array = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
            
            # 解析相机位姿
            transform_matrix = np.array(frame['transform_matrix'])
            
            # OpenGL到OpenCV坐标系转换
            transform_matrix[:, 1:3] *= -1
            
            R = transform_matrix[:3, :3]
            T = transform_matrix[:3, 3]
            
            # 计算FoV
            img_height, img_width = img_array.shape[:2]
            focal = 0.5 * img_width / np.tan(0.5 * camera_angle_x)
            FoVx = camera_angle_x
            FoVy = 2 * np.arctan(img_height / (2 * focal))
            
            cameras.append({
                'uid': i,
                'R': torch.from_numpy(R).float(),
                'T': torch.from_numpy(T).float(),
                'FoVx': FoVx,
                'FoVy': FoVy,
                'image': img_tensor,
                'image_name': img_name,
                'width': img_width,
                'height': img_height
            })
        
        return cameras
    
    def load_point_cloud(self) -> np.ndarray:
        """
        合成数据通常没有初始点云，返回空数组
        """
        return np.array([])


def create_synthetic_scene(num_cameras=10, image_size=(256, 256)):
    """
    创建一个简单的合成场景用于测试
    """
    cameras = []
    h, w = image_size
    
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        radius = 3.0
        
        # 相机位置
        cam_pos = np.array([
            radius * np.cos(angle),
            0.0,
            radius * np.sin(angle)
        ])
        
        # 朝向原点
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        R = np.stack([right, up, forward], axis=1)
        T = cam_pos
        
        # 创建随机图像
        img_tensor = torch.rand(3, h, w)
        
        cameras.append({
            'uid': i,
            'R': torch.from_numpy(R).float(),
            'T': torch.from_numpy(T).float(),
            'FoVx': 0.691111,
            'FoVy': 0.691111,
            'image': img_tensor,
            'image_name': f'test_{i:03d}.png',
            'width': w,
            'height': h
        })
    
    return cameras
