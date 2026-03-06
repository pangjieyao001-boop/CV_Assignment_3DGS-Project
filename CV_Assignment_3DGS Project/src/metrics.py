"""
评估指标计算
PSNR, SSIM, LPIPS
"""

import torch
import numpy as np
from typing import Union
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips not available, LPIPS metric will not be computed")


def compute_psnr(img1: Union[torch.Tensor, np.ndarray], 
                 img2: Union[torch.Tensor, np.ndarray],
                 data_range: float = 1.0) -> float:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 预测图像 [H, W, 3] 或 [3, H, W]
        img2: 真实图像 [H, W, 3] 或 [3, H, W]
        data_range: 数据范围 (1.0 for normalized images, 255 for uint8)
    
    Returns:
        PSNR值 (dB)
    """
    # 转换为numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 确保相同shape
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    
    # 如果通道在第一个维度，调整顺序
    if img1.shape[0] == 3 and len(img1.shape) == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    
    # 确保是float类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    return psnr(img1, img2, data_range=data_range)


def compute_ssim(img1: Union[torch.Tensor, np.ndarray],
                 img2: Union[torch.Tensor, np.ndarray],
                 data_range: float = 1.0) -> float:
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1: 预测图像 [H, W, 3] 或 [3, H, W]
        img2: 真实图像 [H, W, 3] 或 [3, H, W]
        data_range: 数据范围
    
    Returns:
        SSIM值 [-1, 1]，越接近1越好
    """
    # 转换为numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 确保相同shape
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    
    # 如果通道在第一个维度，调整顺序
    if img1.shape[0] == 3 and len(img1.shape) == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    
    # 确保是float类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算SSIM
    return ssim(img1, img2, data_range=data_range, channel_axis=2)


def compute_lpips(img1: Union[torch.Tensor, np.ndarray],
                  img2: Union[torch.Tensor, np.ndarray],
                  net='alex', device='cpu') -> float:
    """
    计算LPIPS (Learned Perceptual Image Patch Similarity)
    
    Args:
        img1: 预测图像 [H, W, 3] 或 [3, H, W]，值范围[-1, 1]或[0, 1]
        img2: 真实图像 [H, W, 3] 或 [3, H, W]
        net: 使用的网络 ('alex', 'vgg', 'squeeze')
        device: 计算设备
    
    Returns:
        LPIPS值，越低越好
    """
    if not LPIPS_AVAILABLE:
        return 0.0
    
    # 转换为tensor
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # 确保通道在第一个维度
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        img1 = img1.permute(2, 0, 1)
    if img2.shape[-1] == 3 and len(img2.shape) == 3:
        img2 = img2.permute(2, 0, 1)
    
    # 添加batch维度
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # 确保值范围在[-1, 1]
    if img1.max() <= 1.0:
        img1 = img1 * 2 - 1
    if img2.max() <= 1.0:
        img2 = img2 * 2 - 1
    
    # 初始化LPIPS模型
    loss_fn = lpips.LPIPS(net=net).to(device)
    
    with torch.no_grad():
        dist = loss_fn(img1.to(device), img2.to(device))
    
    return dist.item()


class MetricsTracker:
    """
    评估指标跟踪器
    """
    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }
    
    def update(self, pred_img, gt_img, compute_lpips_metric=True, device='cpu'):
        """
        更新指标
        """
        psnr_val = compute_psnr(pred_img, gt_img)
        ssim_val = compute_ssim(pred_img, gt_img)
        
        self.metrics['psnr'].append(psnr_val)
        self.metrics['ssim'].append(ssim_val)
        
        if compute_lpips_metric and LPIPS_AVAILABLE:
            lpips_val = compute_lpips(pred_img, gt_img, device=device)
            self.metrics['lpips'].append(lpips_val)
    
    def get_average(self):
        """
        获取平均指标
        """
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        return avg_metrics
    
    def reset(self):
        """
        重置指标
        """
        for key in self.metrics:
            self.metrics[key] = []
    
    def print_summary(self):
        """
        打印指标摘要
        """
        avg_metrics = self.get_average()
        print("\n" + "="*50)
        print("Evaluation Metrics Summary")
        print("="*50)
        for key in ['psnr', 'ssim', 'lpips']:
            if key in avg_metrics:
                mean_val = avg_metrics[key]
                std_val = avg_metrics.get(f'{key}_std', 0)
                print(f"{key.upper():10s}: {mean_val:.4f} ± {std_val:.4f}")
        print("="*50)


def evaluate_model(model, test_cameras, device='cpu'):
    """
    在测试集上评估模型
    
    Args:
        model: Gaussian模型
        test_cameras: 测试相机列表
        device: 计算设备
    
    Returns:
        平均指标字典
    """
    tracker = MetricsTracker()
    
    model.eval()
    with torch.no_grad():
        for cam in test_cameras:
            # 渲染图像（这里需要实现实际的渲染逻辑）
            # rendered = render(model, cam, device)
            # tracker.update(rendered, cam['image'], device=device)
            pass
    
    return tracker.get_average()
