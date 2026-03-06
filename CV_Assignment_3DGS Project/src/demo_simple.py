#!/usr/bin/env python3
"""
简化演示脚本
演示压缩算法的核心功能，无需完整数据集
"""

import torch
import numpy as np
from pathlib import Path
import json

from train_3dgs import GaussianModel
from compression.pruning import GaussianPruner, compute_model_size_mb
from compression.quantization import GaussianQuantizer, SHDistiller
from utils.visualization import plot_metrics_comparison, plot_compression_ratio_comparison


def create_test_model(num_points=10000, device='cpu'):
    """创建一个测试用的Gaussian模型"""
    print(f"Creating test model with {num_points} Gaussians...")
    model = GaussianModel(num_points=num_points, sh_degree=3)
    model = model.to(device)
    return model


def demo_pruning():
    """演示剪枝算法"""
    print("\n" + "="*60)
    print("Demo: Gaussian Pruning")
    print("="*60)
    
    # 创建模型
    model = create_test_model(num_points=10000)
    original_size = compute_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original #Gaussians: {model.get_xyz.shape[0]}")
    
    # 剪枝
    pruner = GaussianPruner(method='opacity', threshold=0.5)
    
    # 模拟一些低不透明度的Gaussians
    with torch.no_grad():
        model._opacity[:3000] = torch.tensor([-5.0])  # 这些会被剪掉
    
    pruned_model = pruner.prune(model)
    
    pruned_size = compute_model_size_mb(pruned_model)
    compression_ratio = original_size / pruned_size
    
    print(f"\nAfter pruning:")
    print(f"  Model size: {pruned_size:.2f} MB")
    print(f"  #Gaussians: {pruned_model.get_xyz.shape[0]}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    return {
        'method': 'Pruning',
        'original_size': original_size,
        'compressed_size': pruned_size,
        'compression_ratio': compression_ratio
    }


def demo_quantization():
    """演示量化算法"""
    print("\n" + "="*60)
    print("Demo: Vector Quantization")
    print("="*60)
    
    # 创建模型
    model = create_test_model(num_points=10000)
    original_size = compute_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # 量化
    quantizer = GaussianQuantizer(config={
        'xyz': {'n_clusters': 256, 'enabled': True},
        'features_dc': {'n_clusters': 256, 'enabled': True},
        'features_rest': {'n_clusters': 128, 'enabled': True},
        'opacity': {'n_clusters': 128, 'enabled': True},
        'scaling': {'n_clusters': 256, 'enabled': True},
        'rotation': {'n_clusters': 256, 'enabled': True},
    })
    
    quantized_state, codebooks = quantizer.quantize_model(model)
    
    # 估算压缩后大小
    num_gaussians = model.get_xyz.shape[0]
    indices_size_mb = num_gaussians * 6 * 2 / 1024 / 1024  # 6 attributes, 2 bytes each
    codebook_size_mb = sum([cb.nbytes for cb in codebooks.values()]) / 1024 / 1024
    quantized_size = indices_size_mb + codebook_size_mb
    
    compression_ratio = original_size / max(quantized_size, 0.01)
    
    print(f"\nAfter quantization:")
    print(f"  Estimated size: {quantized_size:.2f} MB")
    print(f"  Indices: {indices_size_mb:.2f} MB")
    print(f"  Codebooks: {codebook_size_mb:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    return {
        'method': 'Quantization',
        'original_size': original_size,
        'compressed_size': quantized_size,
        'compression_ratio': compression_ratio
    }


def demo_sh_distillation():
    """演示SH蒸馏"""
    print("\n" + "="*60)
    print("Demo: SH Distillation")
    print("="*60)
    
    # 创建模型
    model = create_test_model(num_points=10000)
    original_size = compute_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original SH degree: {model.max_sh_degree}")
    print(f"Feature rest shape: {model._features_rest.shape}")
    
    # SH蒸馏
    distiller = SHDistiller(target_degree=1)
    distilled_model = distiller.distill(model)
    
    distilled_size = compute_model_size_mb(distilled_model)
    compression_ratio = original_size / max(distilled_size, 0.01)
    
    print(f"\nAfter SH distillation:")
    print(f"  Model size: {distilled_size:.2f} MB")
    print(f"  SH degree: {distilled_model.max_sh_degree}")
    print(f"  Feature rest shape: {distilled_model._features_rest.shape}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    return {
        'method': 'SH Distillation',
        'original_size': original_size,
        'compressed_size': distilled_size,
        'compression_ratio': compression_ratio
    }


def demo_hybrid():
    """演示混合压缩"""
    print("\n" + "="*60)
    print("Demo: Hybrid Compression (Pruning + Quantization)")
    print("="*60)
    
    # 创建模型
    model = create_test_model(num_points=10000)
    original_size = compute_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # 1. 剪枝
    with torch.no_grad():
        model._opacity[:3000] = torch.tensor([-5.0])
    
    pruner = GaussianPruner(method='opacity', threshold=0.5)
    model = pruner.prune(model)
    print(f"After pruning: {model.get_xyz.shape[0]} Gaussians")
    
    # 2. 量化
    quantizer = GaussianQuantizer(config={
        'xyz': {'n_clusters': 256, 'enabled': True},
        'features_dc': {'n_clusters': 256, 'enabled': True},
        'features_rest': {'n_clusters': 128, 'enabled': True},
        'opacity': {'n_clusters': 128, 'enabled': True},
        'scaling': {'n_clusters': 256, 'enabled': True},
        'rotation': {'n_clusters': 256, 'enabled': True},
    })
    
    quantized_state, codebooks = quantizer.quantize_model(model)
    
    # 估算大小
    num_gaussians = model.get_xyz.shape[0]
    indices_size_mb = num_gaussians * 6 * 2 / 1024 / 1024
    codebook_size_mb = sum([cb.nbytes for cb in codebooks.values()]) / 1024 / 1024
    hybrid_size = indices_size_mb + codebook_size_mb
    
    compression_ratio = original_size / max(hybrid_size, 0.01)
    
    print(f"\nAfter hybrid compression:")
    print(f"  Estimated size: {hybrid_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    return {
        'method': 'Hybrid',
        'original_size': original_size,
        'compressed_size': hybrid_size,
        'compression_ratio': compression_ratio
    }


def generate_summary_report(results):
    """生成摘要报告"""
    print("\n" + "="*60)
    print("Summary Report")
    print("="*60)
    
    print(f"\n{'Method':<20} {'Original(MB)':<15} {'Compressed(MB)':<15} {'Ratio':<10}")
    print("-"*60)
    
    for result in results:
        print(f"{result['method']:<20} {result['original_size']:<15.2f} "
              f"{result['compressed_size']:<15.2f} {result['compression_ratio']:<10.2f}x")
    
    # 保存JSON报告
    output_path = Path('output')
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / 'demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path / 'demo_results.json'}")
    
    # 绘制对比图
    results_dict = {r['method']: r for r in results}
    
    try:
        plot_compression_ratio_comparison(
            results_dict,
            str(output_path / 'demo_compression_ratio.png')
        )
        print(f"Compression ratio plot saved to: {output_path / 'demo_compression_ratio.png'}")
    except Exception as e:
        print(f"Could not generate plot: {e}")


def main():
    """主函数"""
    print("="*60)
    print("3DGS Compression Algorithms Demo")
    print("="*60)
    print("\nThis demo shows the compression capabilities without requiring")
    print("a full dataset or lengthy training process.")
    
    results = []
    
    # 运行各个演示
    results.append(demo_pruning())
    results.append(demo_quantization())
    results.append(demo_sh_distillation())
    results.append(demo_hybrid())
    
    # 生成报告
    generate_summary_report(results)
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Download a real dataset (see QUICKSTART.md)")
    print("2. Run: python run_experiments.py -s data/bicycle -m output/exp --full")
    print("3. Check the results in the output/exp/ directory")


if __name__ == "__main__":
    main()
