#!/usr/bin/env python3
"""
从已有实验数据生成报告可视化
不依赖模型渲染，直接使用final_results.json中的数据
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output/report_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载已有数据
with open("final_results.json", "r") as f:
    results = json.load(f)

print("Loaded results:")
for r in results:
    print(f"  {r['name']}: {r['ratio']:.2f}x compression, PSNR: {r['psnr_mean']:.2f}")


def create_compression_summary_chart(results, output_path):
    """创建压缩综合对比图"""
    fig = plt.figure(figsize=(16, 10))
    
    names = [r['name'] for r in results]
    ratios = [r['ratio'] for r in results]
    psnrs = [r['psnr_mean'] for r in results]
    ssims = [r['ssim_mean'] for r in results]
    sizes = [r['size_mb'] for r in results]
    gaussians = [r['gaussians'] for r in results]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    # 1. 压缩率
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(range(len(names)), ratios, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Compression Ratio by Method', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for i, r in enumerate(ratios):
        ax1.text(i, r + 0.1, f'{r:.2f}×', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # 2. 模型大小
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(range(len(names)), sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for i, s in enumerate(sizes):
        ax2.text(i, s + 0.05, f'{s:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. PSNR对比
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(range(len(names)), psnrs, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax3.set_title('PSNR Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax3.axhline(y=psnrs[0], color='red', linestyle='--', linewidth=2, label='Baseline')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Rate-Distortion曲线
    ax4 = plt.subplot(2, 3, 4)
    # 按压缩率排序
    sorted_idx = np.argsort(ratios)
    sorted_ratios = [ratios[i] for i in sorted_idx]
    sorted_psnrs = [psnrs[i] for i in sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]
    
    ax4.plot(sorted_ratios, sorted_psnrs, 'o-', linewidth=3, markersize=12, 
            color='steelblue', label='Rate-Distortion')
    ax4.scatter([ratios[0]], [psnrs[0]], s=400, c='red', marker='*', 
               label='Baseline', zorder=5, edgecolors='black', linewidths=2)
    for i, name in enumerate(sorted_names):
        ax4.annotate(name, (sorted_ratios[i], sorted_psnrs[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax4.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax4.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax4.set_title('Rate-Distortion Curve', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    # 5. SSIM对比
    ax5 = plt.subplot(2, 3, 5)
    bars5 = ax5.bar(range(len(names)), ssims, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax5.set_title('SSIM Comparison', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(len(names)))
    ax5.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax5.set_ylim([0.6, 0.7])
    ax5.axhline(y=ssims[0], color='red', linestyle='--', linewidth=2, label='Baseline')
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. 高斯数量
    ax6 = plt.subplot(2, 3, 6)
    bars6 = ax6.bar(range(len(names)), gaussians, color=colors, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Number of Gaussians', fontsize=12, fontweight='bold')
    ax6.set_title('Gaussians by Method', fontsize=13, fontweight='bold')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax6.axhline(y=gaussians[0], color='red', linestyle='--', linewidth=2, label='Baseline')
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    for i, g in enumerate(gaussians):
        ax6.text(i, g + 100, f'{g}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('3DGS Compression - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_sh_degree_analysis(results, output_path):
    """SH阶数详细分析"""
    # 提取SH相关方法
    sh_methods = [r for r in results if 'SH_deg' in r['name']]
    
    if len(sh_methods) < 3:
        print("Not enough SH methods found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['name'].replace('SH_deg_', 'Degree ') for r in sh_methods]
    ratios = [r['ratio'] for r in sh_methods]
    psnrs = [r['psnr_mean'] for r in sh_methods]
    ssims = [r['ssim_mean'] for r in sh_methods]
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))
    
    # 压缩率
    ax = axes[0]
    bars = ax.bar(names, ratios, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Compression by SH Degree', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.1, 
                f'{val:.2f}×', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # PSNR
    ax = axes[1]
    ax.plot(names, psnrs, 'o-', linewidth=3, markersize=15, color='steelblue')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('PSNR by SH Degree', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([10.5, 11.0])
    
    # SSIM
    ax = axes[2]
    ax.plot(names, ssims, 's-', linewidth=3, markersize=15, color='coral')
    ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax.set_title('SSIM by SH Degree', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0.65, 0.70])
    
    plt.suptitle('Spherical Harmonics Degree Reduction Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_method_comparison_table(results, output_path):
    """创建方法对比表格图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    for r in results:
        reduction = (1 - 1/r['ratio']) * 100 if r['ratio'] > 0 else 0
        table_data.append([
            r['name'],
            f"{r['size_mb']:.2f}",
            f"{r['ratio']:.2f}×",
            f"{reduction:.1f}%",
            f"{r['gaussians']}",
            f"{r['psnr_mean']:.2f}±{r['psnr_std']:.2f}",
            f"{r['ssim_mean']:.4f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'Size (MB)', 'Ratio', 'Reduction', 'Gaussians', 'PSNR (dB)', 'SSIM'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0.1, 1, 0.8]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(7):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # 设置最佳结果的行颜色
    best_idx = max(range(len(results)), key=lambda i: results[i]['ratio'])
    for i in range(7):
        table[(best_idx + 1, i)].set_facecolor('#A23B72')
        table[(best_idx + 1, i)].set_text_props(weight='bold', color='white')
    
    # 设置基线行颜色
    for i in range(7):
        table[(1, i)].set_facecolor('#F18F01')
        table[(1, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Compression Methods Comparison Table\n(Yellow=Baseline, Purple=Best)',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_size_reduction_visualization(results, output_path):
    """创建大小减少可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 排除baseline
    compressed = [r for r in results if r['name'] != 'Baseline']
    
    names = [r['name'] for r in compressed]
    reductions = [(1 - 1/r['ratio']) * 100 for r in compressed]
    sizes = [r['size_mb'] for r in compressed]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))[::-1]
    
    # 1. 大小减少百分比
    ax = axes[0]
    bars = ax.barh(names, reductions, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Size Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Size Reduction vs Baseline', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, reductions):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', ha='left', va='center', 
                fontsize=10, fontweight='bold')
    
    # 2. 文件大小对比
    ax = axes[1]
    baseline_size = results[0]['size_mb']
    x_pos = np.arange(len(names))
    
    bars1 = ax.bar(x_pos - 0.2, [baseline_size] * len(names), 0.4, 
                   label='Baseline', color='coral', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + 0.2, sizes, 0.4, 
                   label='Compressed', color='steelblue', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Size (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Size Comparison: Baseline vs Compressed', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Size Reduction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("="*60)
    print("Generating Report Visualizations from Data")
    print("="*60)
    
    # 1. 综合对比图
    create_compression_summary_chart(
        results, 
        OUTPUT_DIR / 'comprehensive_analysis.png'
    )
    
    # 2. SH阶数分析
    create_sh_degree_analysis(
        results,
        OUTPUT_DIR / 'sh_degree_analysis.png'
    )
    
    # 3. 方法对比表
    create_method_comparison_table(
        results,
        OUTPUT_DIR / 'method_comparison_table.png'
    )
    
    # 4. 大小减少可视化
    create_size_reduction_visualization(
        results,
        OUTPUT_DIR / 'size_reduction.png'
    )
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("="*60)
    print("\nFiles generated:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
