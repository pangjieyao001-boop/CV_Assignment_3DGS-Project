"""
Gaussian剪枝算法
基于不透明度和梯度的重要度剪枝
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class GaussianPruner:
    """
    Gaussian剪枝器
    支持多种剪枝策略
    """
    
    def __init__(self, method='opacity', threshold=0.005):
        """
        Args:
            method: 剪枝方法 ('opacity', 'gradient', 'combined')
            threshold: 剪枝阈值
        """
        self.method = method
        self.threshold = threshold
    
    def prune_by_opacity(self, model, return_mask=False):
        """
        基于不透明度的剪枝
        
        Args:
            model: Gaussian模型
            return_mask: 是否返回掩码
        
        Returns:
            剪枝后的模型或(模型, 掩码)
        """
        with torch.no_grad():
            opacities = model.get_opacity
            prune_mask = opacities.squeeze(-1) < self.threshold
            
            n_pruned = prune_mask.sum().item()
            n_total = len(opacities)
            
            if n_pruned > 0:
                model.prune_points(prune_mask)
                print(f"Pruned {n_pruned}/{n_total} Gaussians ({100*n_pruned/n_total:.2f}%)")
            
            if return_mask:
                return model, prune_mask
            return model
    
    def prune_by_gradient(self, model, gradients, return_mask=False):
        """
        基于梯度的重要度剪枝
        
        Args:
            model: Gaussian模型
            gradients: 位置梯度
            return_mask: 是否返回掩码
        
        Returns:
            剪枝后的模型或(模型, 掩码)
        """
        with torch.no_grad():
            # 计算梯度范数
            grad_norms = torch.norm(gradients, dim=-1)
            prune_mask = grad_norms < self.threshold
            
            n_pruned = prune_mask.sum().item()
            n_total = len(grad_norms)
            
            if n_pruned > 0:
                model.prune_points(prune_mask)
                print(f"Pruned {n_pruned}/{n_total} Gaussians by gradient ({100*n_pruned/n_total:.2f}%)")
            
            if return_mask:
                return model, prune_mask
            return model
    
    def prune_by_size(self, model, scene_extent, size_threshold=0.01, return_mask=False):
        """
        基于大小的剪枝（移除过大的Gaussians）
        
        Args:
            model: Gaussian模型
            scene_extent: 场景范围
            size_threshold: 大小阈值（相对于场景范围的比例）
            return_mask: 是否返回掩码
        
        Returns:
            剪枝后的模型或(模型, 掩码)
        """
        with torch.no_grad():
            scales = model.get_scaling
            max_scales = torch.max(scales, dim=1).values
            
            prune_mask = max_scales > scene_extent * size_threshold
            
            n_pruned = prune_mask.sum().item()
            n_total = len(scales)
            
            if n_pruned > 0:
                model.prune_points(prune_mask)
                print(f"Pruned {n_pruned}/{n_total} large Gaussians ({100*n_pruned/n_total:.2f}%)")
            
            if return_mask:
                return model, prune_mask
            return model
    
    def prune(self, model, **kwargs):
        """
        执行剪枝
        """
        if self.method == 'opacity':
            return self.prune_by_opacity(model, **kwargs)
        elif self.method == 'gradient':
            return self.prune_by_gradient(model, **kwargs)
        elif self.method == 'size':
            return self.prune_by_size(model, **kwargs)
        else:
            raise ValueError(f"Unknown pruning method: {self.method}")


class AdaptivePruner(GaussianPruner):
    """
    自适应剪枝器
    根据场景复杂度动态调整剪枝策略
    """
    
    def __init__(self, target_compression_ratio=0.5, max_iterations=10):
        super().__init__()
        self.target_compression_ratio = target_compression_ratio
        self.max_iterations = max_iterations
    
    def adaptive_prune(self, model, scene_extent):
        """
        自适应剪枝，达到目标压缩率
        """
        initial_count = model.get_xyz.shape[0]
        target_count = int(initial_count * (1 - self.target_compression_ratio))
        
        current_threshold = 0.005
        best_model = None
        best_count = initial_count
        
        for iteration in range(self.max_iterations):
            # 复制模型
            test_model = model  # 实际操作中应该深拷贝
            
            # 尝试剪枝
            self.threshold = current_threshold
            pruned_model = self.prune_by_opacity(test_model)
            
            current_count = pruned_model.get_xyz.shape[0]
            current_ratio = 1 - current_count / initial_count
            
            print(f"Iteration {iteration}: threshold={current_threshold:.4f}, "
                  f"compressed {100*current_ratio:.2f}%, target {100*self.target_compression_ratio:.2f}%")
            
            if abs(current_ratio - self.target_compression_ratio) < 0.05:
                # 达到目标
                return pruned_model
            
            if current_count < target_count:
                # 剪枝太多，降低阈值
                current_threshold *= 0.8
            else:
                # 剪枝太少，增加阈值
                current_threshold *= 1.2
                best_model = pruned_model
                best_count = current_count
        
        return best_model if best_model else model


def compute_compression_ratio(original_model, compressed_model):
    """
    计算压缩比
    
    Args:
        original_model: 原始模型
        compressed_model: 压缩后的模型
    
    Returns:
        压缩比 (原始大小 / 压缩后大小)
    """
    orig_params = sum(p.numel() for p in original_model.parameters())
    comp_params = sum(p.numel() for p in compressed_model.parameters())
    
    return orig_params / comp_params


def compute_model_size_mb(model):
    """
    计算模型大小（MB）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb
