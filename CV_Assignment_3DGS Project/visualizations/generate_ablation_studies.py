#!/usr/bin/env python3
"""
生成消融研究可视化
包括：SH逐阶分析、混合策略对比、训练动态等
"""

import sys
sys.path.insert(0, './3dgs_project')

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_3dgs import GaussianModel
from compression.pruning import GaussianPruner
from compression.quantization import SHDistiller
from utils.data_loader import SyntheticDataLoader
from utils.metrics import compute_psnr, compute_ssim

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = Path("output/ablation_studies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sh_degree_ablation_study(model_path, data_path, output_path):
    """
    SH阶数消融研究
    分析从degree 3逐步降到degree 0的影响
    """
    print("\n" + "="*60)
    print("SH Degree Ablation Study")
    print("="*60)
    
    # 加载基线模型
    checkpoint = torch.load(Path(model_path) / 'model_final.pth', map_location='cpu')
    config = json.load(open(Path(model_path) / 'config.json'))
    state_dict = checkpoint['model_state_dict']
    
    num_points = state_dict['_xyz'].shape[0]
    baseline_model = GaussianModel(num_points, 3)
    baseline_model._xyz = torch.nn.Parameter(state_dict['_xyz'])
    baseline_model._features_dc = torch.nn.Parameter(state_dict['_features_dc'])
    baseline_model._features_rest = torch.nn.Parameter(state_dict['_features_rest'])
    baseline_model._opacity = torch.nn.Parameter(state_dict['_opacity'])
    baseline_model._scaling = torch.nn.Parameter(state_dict['_scaling'])
    baseline_model._rotation = torch.nn.Parameter(state_dict['_rotation'])
    baseline_model = baseline_model.to(DEVICE)
    
    # 加载测试数据
    loader = SyntheticDataLoader(data_path, device=DEVICE)
    test_cameras = loader.load_cameras('test')
    
    # 测试不同SH degree
    degrees = [3, 2, 1, 0]
    results = []
    
    for deg in degrees:
        print(f"\nTesting SH degree {deg}...")
        
        # 复制模型并蒸馏
        model = GaussianModel(num_points, 3)
        model._xyz = torch.nn.Parameter(state_dict['_xyz'].clone())
        model._features_dc = torch.nn.Parameter(state_dict['_features_dc'].clone())
        model._features_rest = torch.nn.Parameter(state_dict['_features_rest'].clone())
        model._opacity = torch.nn.Parameter(state_dict['_opacity'].clone())
        model._scaling = torch.nn.Parameter(state_dict['_scaling'].clone())
        model._rotation = torch.nn.Parameter(state_dict['_rotation'].clone())
        model = model.to(DEVICE)
        
        if deg < 3:
            distiller = SHDistiller(target_degree=deg)
            distiller.distill(model)
        
        # 计算模型大小
        total_params = sum(p.numel() for p in [model._xyz, model._features_dc, 
                                                model._features_rest, model._opacity,
                                                model._scaling, model._rotation])
        size_mb = total_params * 4 / (1024 * 1024)
        
        # 评估
        psnrs, ssims = [], []
        model.eval()
        
        with torch.no_grad():
            for cam in tqdm(test_cameras[:100], desc=f"Eval SH{deg}", leave=False):
                gt = cam['image'].to(DEVICE)
                rendered = model.render_from_camera(cam, DEVICE)
                
                psnr = compute_psnr(rendered, gt)
                ssim = compute_ssim(rendered, gt)
                
                psnrs.append(psnr)
                ssims.append(ssim)
        
        results.append({
            'degree': deg,
            'coefficients': 3 * (deg + 1) ** 2,
            'size_mb': size_mb,
            'psnr_mean': np.mean(psnrs),
            'psnr_std': np.std(psnrs),
            'ssim_mean': np.mean(ssims),
            'ssim_std': np.std(ssims),
        })
        
        print(f"  SH deg {deg}: {3 * (deg + 1) ** 2} coefficients, "
              f"{size_mb:.2f} MB, PSNR: {np.mean(psnrs):.2f} dB")
    
    # 计算baseline大小
    baseline_params = sum(p.numel() for p in [baseline_model._xyz, baseline_model._features_dc, 
                                               baseline_model._features_rest, baseline_model._opacity,
                                               baseline_model._scaling, baseline_model._rotation])
    baseline_size = baseline_params * 4 / (1024 * 1024)
    
    for r in results:
        r['compression_ratio'] = baseline_size / r['size_mb']
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    degrees_labels = [f"Deg {r['degree']}\n({r['coefficients']} coefs)" for r in results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # 1. 系数数量
    ax = axes[0, 0]
    coefs = [r['coefficients'] for r in results]
    bars = ax.bar(degrees_labels, coefs, color=colors)
    ax.set_ylabel('Number of Coefficients')
    ax.set_title('SH Coefficients by Degree', fontweight='bold')
    for bar, val in zip(bars, coefs):
        ax.text(bar.get_x() + bar.get_width()/2., val + 1, 
                str(val), ha='center', va='bottom', fontsize=9)
    
    # 2. 模型大小
    ax = axes[0, 1]
    sizes = [r['size_mb'] for r in results]
    bars = ax.bar(degrees_labels, sizes, color=colors)
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size by SH Degree', fontweight='bold')
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.05, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 压缩率
    ax = axes[0, 2]
    ratios = [r['compression_ratio'] for r in results]
    bars = ax.bar(degrees_labels, ratios, color=colors)
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio by Degree', fontweight='bold')
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.1, 
                f'{val:.2f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. PSNR
    ax = axes[1, 0]
    psnrs = [r['psnr_mean'] for r in results]
    psnr_stds = [r['psnr_std'] for r in results]
    ax.errorbar(range(len(results)), psnrs, yerr=psnr_stds, 
                fmt='o-', linewidth=2, markersize=10, capsize=5, color='steelblue')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(degrees_labels)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR by SH Degree', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=psnrs[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    
    # 5. SSIM
    ax = axes[1, 1]
    ssims = [r['ssim_mean'] for r in results]
    ssim_stds = [r['ssim_std'] for r in results]
    ax.errorbar(range(len(results)), ssims, yerr=ssim_stds, 
                fmt='s-', linewidth=2, markersize=10, capsize=5, color='coral')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(degrees_labels)
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM by SH Degree', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=ssims[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    
    # 6. 综合对比表
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            f"Deg {r['degree']}",
            f"{r['coefficients']}",
            f"{r['size_mb']:.2f} MB",
            f"{r['compression_ratio']:.2f}×",
            f"{r['psnr_mean']:.2f} dB",
            f"{r['ssim_mean']:.4f}"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['SH Degree', 'Coeffs', 'Size', 'Ratio', 'PSNR', 'SSIM'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Table', fontweight='bold', pad=20)
    
    plt.suptitle('Spherical Harmonics Degree Ablation Study', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ SH ablation study saved: {output_path}")
    
    # 保存结果
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def hybrid_strategy_comparison(model_path, data_path, output_path):
    """
    混合策略对比研究
    测试不同的 Pruning + SH 组合
    """
    print("\n" + "="*60)
    print("Hybrid Strategy Comparison")
    print("="*60)
    
    # 加载模型
    checkpoint = torch.load(Path(model_path) / 'model_final.pth', map_location='cpu')
    config = json.load(open(Path(model_path) / 'config.json'))
    state_dict = checkpoint['model_state_dict']
    
    num_points = state_dict['_xyz'].shape[0]
    
    # 加载数据
    loader = SyntheticDataLoader(data_path, device=DEVICE)
    test_cameras = loader.load_cameras('test')
    
    # 测试策略
    strategies = [
        ('Baseline', None, None),
        ('SH deg 2', None, 2),
        ('SH deg 1', None, 1),
        ('SH deg 0', None, 0),
        ('Prune 0.1 + SH0', 0.1, 0),
        ('Prune 0.2 + SH0', 0.2, 0),
        ('Prune 0.3 + SH0', 0.3, 0),
    ]
    
    results = []
    baseline_size = None
    
    for name, prune_th, sh_deg in strategies:
        print(f"\nTesting: {name}...")
        
        # 复制模型
        model = GaussianModel(num_points, 3)
        model._xyz = torch.nn.Parameter(state_dict['_xyz'].clone())
        model._features_dc = torch.nn.Parameter(state_dict['_features_dc'].clone())
        model._features_rest = torch.nn.Parameter(state_dict['_features_rest'].clone())
        model._opacity = torch.nn.Parameter(state_dict['_opacity'].clone())
        model._scaling = torch.nn.Parameter(state_dict['_scaling'].clone())
        model._rotation = torch.nn.Parameter(state_dict['_rotation'].clone())
        model = model.to(DEVICE)
        
        # 应用剪枝
        if prune_th is not None:
            pruner = GaussianPruner(method='opacity', threshold=prune_th)
            model, mask = pruner.prune_by_opacity(model, return_mask=True)
            n_pruned = mask.sum().item()
            print(f"  Pruned {n_pruned} Gaussians")
        
        # 应用SH蒸馏
        if sh_deg is not None and sh_deg < 3:
            distiller = SHDistiller(target_degree=sh_deg)
            distiller.distill(model)
        
        # 计算大小
        total_params = sum(p.numel() for p in [model._xyz, model._features_dc, 
                                                model._features_rest, model._opacity,
                                                model._scaling, model._rotation])
        size_mb = total_params * 4 / (1024 * 1024)
        
        if baseline_size is None:
            baseline_size = size_mb
        
        # 评估
        psnrs, ssims = [], []
        model.eval()
        
        with torch.no_grad():
            for cam in tqdm(test_cameras[:50], desc=f"Eval {name}", leave=False):
                gt = cam['image'].to(DEVICE)
                rendered = model.render_from_camera(cam, DEVICE)
                
                psnr = compute_psnr(rendered, gt)
                ssim = compute_ssim(rendered, gt)
                
                psnrs.append(psnr)
                ssims.append(ssim)
        
        # 避免除零错误
        ratio = baseline_size / size_mb if size_mb > 0 else 0.0
        
        results.append({
            'name': name,
            'prune_threshold': prune_th,
            'sh_degree': sh_deg if sh_deg is not None else 3,
            'size_mb': size_mb,
            'compression_ratio': ratio,
            'gaussians': model._xyz.shape[0],
            'psnr_mean': np.mean(psnrs),
            'psnr_std': np.std(psnrs),
            'ssim_mean': np.mean(ssims),
            'ssim_std': np.std(ssims),
        })
        
        ratio_str = f'{ratio:.2f}×' if size_mb > 0 else 'N/A (all pruned)'
        print(f"  Size: {size_mb:.2f} MB, Ratio: {ratio_str}, "
              f"PSNR: {np.mean(psnrs):.2f} dB")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r['name'] for r in results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # 1. 压缩率对比
    ax = axes[0, 0]
    ratios = [r['compression_ratio'] for r in results]
    bars = ax.bar(names, ratios, color=colors)
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio by Strategy', fontweight='bold')
    ax.set_xticklabels(names, rotation=45, ha='right')
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.1, 
                f'{val:.2f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. PSNR对比
    ax = axes[0, 1]
    psnrs = [r['psnr_mean'] for r in results]
    ax.bar(names, psnrs, color=colors)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR by Strategy', fontweight='bold')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=psnrs[0], color='r', linestyle='--', alpha=0.5)
    
    # 3. 高斯数量
    ax = axes[1, 0]
    gaussians = [r['gaussians'] for r in results]
    bars = ax.bar(names, gaussians, color=colors)
    ax.set_ylabel('Number of Gaussians')
    ax.set_title('Gaussians by Strategy', fontweight='bold')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=gaussians[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    
    # 4. Rate-Distortion曲线
    ax = axes[1, 1]
    sizes = [r['size_mb'] for r in results]
    ax.scatter(sizes, psnrs, s=300, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    for i, name in enumerate(names):
        ax.annotate(name, (sizes[i], psnrs[i]), 
                   textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Rate-Distortion Curve', fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Hybrid Compression Strategy Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Hybrid strategy comparison saved: {output_path}")
    
    # 保存结果
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """主函数"""
    print("="*60)
    print("Generating Ablation Studies")
    print("="*60)
    
    # 使用Lego场景进行消融研究
    model_path = "3dgs_project/output/high_score/lego"
    data_path = "3dgs_project/data/lego"
    
    if not Path(model_path).exists():
        print(f"\n⚠️  Model not found at {model_path}")
        print("Please train the model first or check the path.")
        return
    
    # 1. SH阶数消融研究
    sh_results = sh_degree_ablation_study(
        model_path, data_path,
        OUTPUT_DIR / 'sh_degree_ablation.png'
    )
    
    # 2. 混合策略对比
    hybrid_results = hybrid_strategy_comparison(
        model_path, data_path,
        OUTPUT_DIR / 'hybrid_strategy_comparison.png'
    )
    
    print("\n" + "="*60)
    print("All ablation studies completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
