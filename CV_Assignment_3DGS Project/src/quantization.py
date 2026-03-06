"""
向量量化算法
基于k-means的码本量化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans


class VectorQuantizer:
    """
    向量量化器
    使用k-means聚类创建码本
    """
    
    def __init__(self, n_clusters=256, max_iter=100):
        """
        Args:
            n_clusters: 码本大小（聚类中心数）
            max_iter: k-means最大迭代次数
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.codebook = None
        self.labels = None
    
    def fit(self, data: np.ndarray):
        """
        训练码本
        
        Args:
            data: 输入数据 [N, D]
        """
        # 使用k-means聚类
        kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(data)),
            max_iter=self.max_iter,
            random_state=42,
            n_init=10
        )
        
        self.labels = kmeans.fit_predict(data)
        self.codebook = kmeans.cluster_centers_
        
        return self
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        编码数据为码本索引
        
        Args:
            data: 输入数据 [N, D]
        
        Returns:
            索引 [N]
        """
        if self.codebook is None:
            raise ValueError("Codebook not trained. Call fit() first.")
        
        # 计算到每个码本向量的距离
        distances = np.linalg.norm(
            data[:, np.newaxis, :] - self.codebook[np.newaxis, :, :],
            axis=2
        )
        
        # 选择最近的码本
        indices = np.argmin(distances, axis=1)
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        解码索引为数据
        
        Args:
            indices: 索引 [N]
        
        Returns:
            解码数据 [N, D]
        """
        if self.codebook is None:
            raise ValueError("Codebook not trained. Call fit() first.")
        
        return self.codebook[indices]
    
    def get_compression_stats(self, original_data: np.ndarray) -> Dict:
        """
        获取压缩统计信息
        """
        n_elements = len(original_data)
        element_size = original_data.dtype.itemsize
        
        # 原始大小
        original_bits = n_elements * original_data.shape[1] * 8
        
        # 压缩后大小（索引 + 码本）
        index_bits = n_elements * np.ceil(np.log2(len(self.codebook)))
        codebook_bits = len(self.codebook) * original_data.shape[1] * 32  # 码本用float32
        
        compressed_bits = index_bits + codebook_bits
        
        compression_ratio = original_bits / compressed_bits
        
        return {
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'bpp': compressed_bits / n_elements,
        }


class GaussianQuantizer:
    """
    针对3DGS的专用量化器
    对不同属性分别进行量化
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: 配置字典，指定各属性的量化参数
        """
        self.config = config or {
            'xyz': {'n_clusters': 1024, 'enabled': True},
            'features_dc': {'n_clusters': 256, 'enabled': True},
            'features_rest': {'n_clusters': 256, 'enabled': True},
            'opacity': {'n_clusters': 128, 'enabled': True},
            'scaling': {'n_clusters': 256, 'enabled': True},
            'rotation': {'n_clusters': 256, 'enabled': True},
        }
        
        self.quantizers = {}
    
    def quantize_model(self, model):
        """
        量化整个模型
        
        Args:
            model: Gaussian模型
        
        Returns:
            量化后的模型状态和码本
        """
        quantized_state = {}
        codebooks = {}
        
        # 量化位置
        if self.config['xyz']['enabled']:
            xyz = model._xyz.detach().cpu().numpy()
            quantizer = VectorQuantizer(n_clusters=self.config['xyz']['n_clusters'])
            quantizer.fit(xyz)
            indices = quantizer.encode(xyz)
            quantized_xyz = quantizer.decode(indices)
            
            model._xyz.data = torch.from_numpy(quantized_xyz).float().to(model._xyz.device)
            
            quantized_state['xyz_indices'] = indices
            codebooks['xyz'] = quantizer.codebook
            self.quantizers['xyz'] = quantizer
        
        # 量化DC特征
        if self.config['features_dc']['enabled']:
            features_dc = model._features_dc.detach().cpu().numpy()
            quantizer = VectorQuantizer(n_clusters=self.config['features_dc']['n_clusters'])
            quantizer.fit(features_dc)
            indices = quantizer.encode(features_dc)
            quantized_features = quantizer.decode(indices)
            
            model._features_dc.data = torch.from_numpy(quantized_features).float().to(model._features_dc.device)
            
            quantized_state['features_dc_indices'] = indices
            codebooks['features_dc'] = quantizer.codebook
            self.quantizers['features_dc'] = quantizer
        
        # 量化不透明度
        if self.config['opacity']['enabled']:
            opacity = model._opacity.detach().cpu().numpy()
            quantizer = VectorQuantizer(n_clusters=self.config['opacity']['n_clusters'])
            quantizer.fit(opacity)
            indices = quantizer.encode(opacity)
            quantized_opacity = quantizer.decode(indices)
            
            model._opacity.data = torch.from_numpy(quantized_opacity).float().to(model._opacity.device)
            
            quantized_state['opacity_indices'] = indices
            codebooks['opacity'] = quantizer.codebook
            self.quantizers['opacity'] = quantizer
        
        # 量化缩放
        if self.config['scaling']['enabled']:
            scaling = model._scaling.detach().cpu().numpy()
            quantizer = VectorQuantizer(n_clusters=self.config['scaling']['n_clusters'])
            quantizer.fit(scaling)
            indices = quantizer.encode(scaling)
            quantized_scaling = quantizer.decode(indices)
            
            model._scaling.data = torch.from_numpy(quantized_scaling).float().to(model._scaling.device)
            
            quantized_state['scaling_indices'] = indices
            codebooks['scaling'] = quantizer.codebook
            self.quantizers['scaling'] = quantizer
        
        # 量化旋转
        if self.config['rotation']['enabled']:
            rotation = model._rotation.detach().cpu().numpy()
            quantizer = VectorQuantizer(n_clusters=self.config['rotation']['n_clusters'])
            quantizer.fit(rotation)
            indices = quantizer.encode(rotation)
            quantized_rotation = quantizer.decode(indices)
            
            model._rotation.data = torch.from_numpy(quantized_rotation).float().to(model._rotation.device)
            
            quantized_state['rotation_indices'] = indices
            codebooks['rotation'] = quantizer.codebook
            self.quantizers['rotation'] = quantizer
        
        return quantized_state, codebooks
    
    def save_compressed(self, path, quantized_state, codebooks):
        """
        保存压缩后的模型
        """
        np.savez_compressed(
            path,
            **{k: v for k, v in quantized_state.items()},
            **{f'codebook_{k}': v for k, v in codebooks.items()}
        )
    
    def load_compressed(self, path):
        """
        加载压缩后的模型
        """
        data = np.load(path)
        
        quantized_state = {}
        codebooks = {}
        
        for key in data.files:
            if key.startswith('codebook_'):
                codebooks[key[9:]] = data[key]
            else:
                quantized_state[key] = data[key]
        
        return quantized_state, codebooks


class SHDistiller:
    """
    球谐函数蒸馏器
    将高阶SH系数蒸馏到低阶
    """
    
    def __init__(self, target_degree=1):
        """
        Args:
            target_degree: 目标SH阶数 (0, 1, 2, or 3)
        """
        self.target_degree = target_degree
        self.original_degree = None
    
    def distill(self, model, teacher_model=None):
        """
        蒸馏SH系数
        
        Args:
            model: 学生模型（低阶SH）
            teacher_model: 教师模型（高阶SH），如果为None则使用model
        """
        if teacher_model is None:
            teacher_model = model
        
        self.original_degree = model.max_sh_degree
        
        # 截断高阶SH系数
        if model.max_sh_degree > self.target_degree:
            # 只保留前(target_degree+1)^2 - 1个系数
            n_features = 3 * (self.target_degree + 1) ** 2 - 3
            
            with torch.no_grad():
                model._features_rest = nn.Parameter(
                    model._features_rest[:, :n_features]
                )
            
            model.max_sh_degree = self.target_degree
            model.sh_degree = self.target_degree
            
            print(f"Distilled SH from degree {self.original_degree} to {self.target_degree}")
            print(f"Feature rest shape: {model._features_rest.shape}")
        
        return model
    
    def compute_distillation_loss(self, student_render, teacher_render):
        """
        计算蒸馏损失
        """
        return torch.nn.functional.mse_loss(student_render, teacher_render)


class MixedPrecisionQuantizer:
    """
    混合精度量化
    对不同属性使用不同精度
    """
    
    def __init__(self, precision_config=None):
        self.precision_config = precision_config or {
            'xyz': 16,           # 16-bit for position
            'features_dc': 8,    # 8-bit for DC features
            'features_rest': 8,  # 8-bit for rest features
            'opacity': 8,        # 8-bit for opacity
            'scaling': 16,       # 16-bit for scaling
            'rotation': 8,       # 8-bit for rotation
        }
    
    def quantize_to_bits(self, tensor, n_bits):
        """
        将tensor量化到指定bit数
        """
        if n_bits == 32:
            return tensor.float()
        elif n_bits == 16:
            return tensor.half()
        elif n_bits == 8:
            # 8-bit量化（量化为int8范围）
            min_val = tensor.min()
            max_val = tensor.max()
            
            scale = (max_val - min_val) / 255.0
            quantized = ((tensor - min_val) / scale).round().clamp(0, 255).byte()
            
            return quantized, scale, min_val
        else:
            raise ValueError(f"Unsupported bit width: {n_bits}")
    
    def dequantize_from_bits(self, quantized, scale, min_val, original_dtype=torch.float32):
        """
        从量化值反量化
        """
        if isinstance(quantized, torch.Tensor) and quantized.dtype == torch.uint8:
            return quantized.float() * scale + min_val
        return quantized.to(original_dtype)
