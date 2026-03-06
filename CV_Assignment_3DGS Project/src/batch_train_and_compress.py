#!/usr/bin/env python3
"""
Batch Training and Compression - High Score Version
Trains multiple scenes, runs comprehensive compression, generates rich visualizations
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import time

from train_3dgs import GaussianModel
from compression.pruning import GaussianPruner
from compression.quantization import GaussianQuantizer, SHDistiller
from utils.data_loader import SyntheticDataLoader
from utils.metrics import compute_psnr, compute_ssim

# Configuration
SCENES = ['lego', 'drums', 'ship', 'hotdog']
ITERATIONS = {
    'lego': 30000,      # Main scene - full training
    'drums': 15000,     # Secondary scenes
    'ship': 15000,
    'hotdog': 15000,
}

class EnhancedTrainer:
    """Enhanced trainer with logging and checkpointing"""
    
    def __init__(self, scene_name, source_path, output_path, iterations):
        self.scene_name = scene_name
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.iterations = iterations
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\n{'='*60}")
        print(f"Training {scene_name.upper()} - {iterations} iterations")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        # Load data
        loader = SyntheticDataLoader(self.source_path, device=self.device)
        self.train_cameras = loader.load_cameras('train')
        self.test_cameras = loader.load_cameras('test')
        print(f"Loaded {len(self.train_cameras)} train, {len(self.test_cameras)} test cameras")
        
        # Initialize model
        self.sh_degree = 3
        self.model = GaussianModel(10000, self.sh_degree).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model._xyz, 'lr': 0.00016},
            {'params': self.model._features_dc, 'lr': 0.0025},
            {'params': self.model._features_rest, 'lr': 0.0025/20},
            {'params': self.model._opacity, 'lr': 0.05},
            {'params': self.model._scaling, 'lr': 0.005},
            {'params': self.model._rotation, 'lr': 0.001},
        ])
        
        # Training log
        self.train_log = []
        
    def ssim_loss(self, img1, img2):
        """SSIM-based loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = torch.nn.functional.avg_pool2d(img1, 11, 1, 5)
        mu2 = torch.nn.functional.avg_pool2d(img2, 11, 1, 5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = torch.nn.functional.avg_pool2d(img1 ** 2, 11, 1, 5) - mu1_sq
        sigma2_sq = torch.nn.functional.avg_pool2d(img2 ** 2, 11, 1, 5) - mu2_sq
        sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, 11, 1, 5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def train(self):
        """Training with detailed logging"""
        print(f"\nStarting training...")
        
        progress_bar = tqdm(range(self.iterations), desc=f"Training {self.scene_name}")
        
        for iteration in range(self.iterations):
            # Random camera
            cam = self.train_cameras[iteration % len(self.train_cameras)]
            gt_image = cam['image'].to(self.device)
            
            # Render
            rendered = self.model.render_from_camera(cam, self.device)
            
            # Loss
            l1_loss = torch.abs(rendered - gt_image).mean()
            ssim_loss_val = self.ssim_loss(rendered.unsqueeze(0), gt_image.unsqueeze(0))
            loss = 0.8 * l1_loss + 0.2 * ssim_loss_val
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log every 100 iterations
            if iteration % 100 == 0:
                self.train_log.append({
                    'iteration': iteration,
                    'loss': loss.item(),
                    'l1': l1_loss.item(),
                    'ssim': ssim_loss_val.item(),
                })
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'l1': f'{l1_loss.item():.4f}'
                })
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Save model
        self.save_model()
        
        # Save training curve
        self.save_training_curve()
        
    def save_model(self):
        """Save trained model"""
        model_dict = {
            'iteration': self.iterations,
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(model_dict, self.output_path / 'model_final.pth')
        
        config = {
            'sh_degree': self.sh_degree,
            'num_points': self.model._xyz.shape[0],
        }
        with open(self.output_path / 'config.json', 'w') as f:
            json.dump(config, f)
        
        print(f"✓ Model saved to {self.output_path}")
    
    def save_training_curve(self):
        """Plot and save training curve"""
        if not self.train_log:
            return
        
        iterations = [log['iteration'] for log in self.train_log]
        losses = [log['loss'] for log in self.train_log]
        l1s = [log['l1'] for log in self.train_log]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(iterations, losses, label='Total Loss', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.scene_name.capitalize()} - Training Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(iterations, l1s, label='L1 Loss', color='orange', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('L1 Loss')
        axes[1].set_title(f'{self.scene_name.capitalize()} - L1 Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'training_curve.png', dpi=150)
        plt.close()
        
        print(f"✓ Training curve saved")


class CompressionEvaluator:
    """Comprehensive compression evaluation"""
    
    def __init__(self, model_path, scene_name):
        self.model_path = Path(model_path)
        self.scene_name = scene_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model
        checkpoint = torch.load(self.model_path / 'model_final.pth', map_location='cpu')
        config = json.load(open(self.model_path / 'config.json'))
        state_dict = checkpoint['model_state_dict']
        
        num_points = state_dict['_xyz'].shape[0]
        sh_degree = config.get('sh_degree', 3)
        
        self.baseline_model = GaussianModel(num_points, sh_degree)
        self.baseline_model._xyz = torch.nn.Parameter(state_dict['_xyz'])
        self.baseline_model._features_dc = torch.nn.Parameter(state_dict['_features_dc'])
        self.baseline_model._features_rest = torch.nn.Parameter(state_dict['_features_rest'])
        self.baseline_model._opacity = torch.nn.Parameter(state_dict['_opacity'])
        self.baseline_model._scaling = torch.nn.Parameter(state_dict['_scaling'])
        self.baseline_model._rotation = torch.nn.Parameter(state_dict['_rotation'])
        self.baseline_model = self.baseline_model.to(self.device)
        
        self.baseline_size = self.calc_size(self.baseline_model)
        
        # Load test data
        data_path = Path("data") / scene_name
        loader = SyntheticDataLoader(data_path, device=self.device)
        self.test_cameras = loader.load_cameras('test')
        
        print(f"\nLoaded {scene_name}: {num_points} Gaussians, {self.baseline_size:.2f} MB")
    
    def calc_size(self, model):
        """Calculate model size in MB"""
        total = sum(p.numel() for p in [model._xyz, model._features_dc, 
                                        model._features_rest, model._opacity,
                                        model._scaling, model._rotation])
        return total * 4 / (1024 * 1024)
    
    def copy_model(self, model):
        """Deep copy model"""
        new_model = GaussianModel(model._xyz.shape[0], model.sh_degree)
        new_model._xyz = torch.nn.Parameter(model._xyz.clone())
        new_model._features_dc = torch.nn.Parameter(model._features_dc.clone())
        new_model._features_rest = torch.nn.Parameter(model._features_rest.clone())
        new_model._opacity = torch.nn.Parameter(model._opacity.clone())
        new_model._scaling = torch.nn.Parameter(model._scaling.clone())
        new_model._rotation = torch.nn.Parameter(model._rotation.clone())
        return new_model.to(self.device)
    
    def evaluate(self, model, name="Model", sample_size=50):
        """Evaluate model quality"""
        model.eval()
        psnrs, ssims = [], []
        
        # Sample cameras for speed
        sample_cameras = self.test_cameras[:sample_size]
        
        with torch.no_grad():
            for cam in tqdm(sample_cameras, desc=f"Eval {name}", leave=False):
                gt = cam['image'].to(self.device)
                rendered = model.render_from_camera(cam, self.device)
                
                psnr = compute_psnr(rendered, gt)
                ssim = compute_ssim(rendered, gt)
                
                psnrs.append(psnr)
                ssims.append(ssim)
        
        return {
            'psnr_mean': np.mean(psnrs),
            'psnr_std': np.std(psnrs),
            'ssim_mean': np.mean(ssims),
            'ssim_std': np.std(ssims),
        }
    
    def run_all_experiments(self):
        """Run comprehensive compression experiments"""
        print(f"\n{'='*60}")
        print(f"Compression Experiments - {self.scene_name.upper()}")
        print(f"{'='*60}")
        
        results = []
        
        # Baseline
        print("\n[1/8] Baseline")
        baseline_metrics = self.evaluate(self.baseline_model, "Baseline")
        results.append({
            'name': 'Baseline',
            'size_mb': self.baseline_size,
            'ratio': 1.0,
            'gaussians': self.baseline_model._xyz.shape[0],
            **baseline_metrics
        })
        
        # SH Distillation
        for deg, label in [(2, '2'), (1, '1'), (0, '0')]:
            print(f"\n[{len(results)+1}/8] SH Distillation degree {deg}")
            model = self.copy_model(self.baseline_model)
            distiller = SHDistiller(target_degree=deg)
            distiller.distill(model)
            
            metrics = self.evaluate(model, f"SH deg {deg}")
            results.append({
                'name': f'SH_deg_{label}',
                'size_mb': self.calc_size(model),
                'ratio': self.baseline_size / self.calc_size(model),
                'gaussians': model._xyz.shape[0],
                **metrics
            })
        
        # Pruning + SH Distillation
        for th in [0.05, 0.1, 0.2]:
            print(f"\n[{len(results)+1}/8] Prune (th={th}) + SH deg 0")
            model = self.copy_model(self.baseline_model)
            
            pruner = GaussianPruner(method='opacity', threshold=th)
            pruned, _ = pruner.prune_by_opacity(model, return_mask=True)
            
            distiller = SHDistiller(target_degree=0)
            distiller.distill(pruned)
            
            metrics = self.evaluate(pruned, f"Prune {th}")
            results.append({
                'name': f'Prune{th}_SH0',
                'size_mb': self.calc_size(pruned),
                'ratio': self.baseline_size / self.calc_size(pruned),
                'gaussians': pruned._xyz.shape[0],
                **metrics
            })
        
        return results
    
    def visualize(self, results, output_dir):
        """Create comprehensive visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        names = [r['name'] for r in results]
        ratios = [r['ratio'] for r in results]
        psnrs = [r['psnr_mean'] for r in results]
        ssims = [r['ssim_mean'] for r in results]
        sizes = [r['size_mb'] for r in results]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Compression Ratio
        ax1 = plt.subplot(2, 3, 1)
        bars1 = ax1.bar(range(len(names)), ratios, color=colors)
        ax1.set_ylabel('Compression Ratio', fontsize=11)
        ax1.set_title(f'{self.scene_name.capitalize()} - Compression Ratios', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        for i, r in enumerate(ratios):
            ax1.text(i, r + 0.1, f'{r:.2f}×', ha='center', va='bottom', fontsize=8)
        
        # 2. PSNR
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(range(len(names)), psnrs, color=colors)
        ax2.set_ylabel('PSNR (dB)', fontsize=11)
        ax2.set_title(f'{self.scene_name.capitalize()} - PSNR Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax2.axhline(y=results[0]['psnr_mean'], color='r', linestyle='--', label='Baseline')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. SSIM
        ax3 = plt.subplot(2, 3, 3)
        bars3 = ax3.bar(range(len(names)), ssims, color=colors)
        ax3.set_ylabel('SSIM', fontsize=11)
        ax3.set_title(f'{self.scene_name.capitalize()} - SSIM Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax3.axhline(y=results[0]['ssim_mean'], color='r', linestyle='--', label='Baseline')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. PSNR vs Size (Scatter)
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(sizes, psnrs, s=300, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
        ax4.axvline(x=self.baseline_size, color='r', linestyle='--', alpha=0.5, label=f'Baseline ({self.baseline_size:.2f} MB)')
        for i, name in enumerate(names):
            ax4.annotate(name, (sizes[i], psnrs[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax4.set_xlabel('Model Size (MB)', fontsize=11)
        ax4.set_ylabel('PSNR (dB)', fontsize=11)
        ax4.set_title('PSNR vs Model Size Trade-off', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Rate-Distortion Curve
        ax5 = plt.subplot(2, 3, 5)
        sorted_results = sorted(results, key=lambda x: x['ratio'])
        ax5.plot([r['ratio'] for r in sorted_results], [r['psnr_mean'] for r in sorted_results], 
                'o-', linewidth=2.5, markersize=10, color='steelblue', label='Rate-Distortion')
        ax5.scatter([results[0]['ratio']], [results[0]['psnr_mean']], s=300, c='red', marker='*', 
                   label='Baseline', zorder=5, edgecolors='black', linewidths=2)
        for r in sorted_results:
            ax5.annotate(r['name'], (r['ratio'], r['psnr_mean']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax5.set_xlabel('Compression Ratio', fontsize=11)
        ax5.set_ylabel('PSNR (dB)', fontsize=11)
        ax5.set_title('Rate-Distortion Curve', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Size Reduction
        ax6 = plt.subplot(2, 3, 6)
        reduction = [(1 - 1/r) * 100 for r in ratios[1:]]  # Exclude baseline
        bars6 = ax6.bar(range(len(names)-1), reduction, color=colors[1:])
        ax6.set_ylabel('Size Reduction (%)', fontsize=11)
        ax6.set_title('Size Reduction vs Baseline', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(len(names)-1))
        ax6.set_xticklabels(names[1:], rotation=45, ha='right', fontsize=9)
        ax6.grid(axis='y', alpha=0.3)
        for i, r in enumerate(reduction):
            ax6.text(i, r + 1, f'{r:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.scene_name}_compression_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {output_dir}")


def main():
    """Main batch processing"""
    all_results = {}
    
    # Train and evaluate each scene
    for scene in SCENES:
        data_path = f"data/{scene}"
        output_path = f"output/high_score/{scene}"
        
        if not Path(data_path).exists():
            print(f"Skipping {scene} - data not found")
            continue
        
        # Training
        print(f"\n{'#'*60}")
        print(f"# Processing {scene.upper()}")
        print(f"{'#'*60}")
        
        trainer = EnhancedTrainer(
            scene_name=scene,
            source_path=data_path,
            output_path=output_path,
            iterations=ITERATIONS[scene]
        )
        
        # Check if already trained
        if not (Path(output_path) / 'model_final.pth').exists():
            trainer.train()
        else:
            print(f"Model already trained, skipping...")
        
        # Compression experiments
        evaluator = CompressionEvaluator(output_path, scene)
        results = evaluator.run_all_experiments()
        evaluator.visualize(results, output_path)
        
        all_results[scene] = results
        
        # Save results
        with open(Path(output_path) / 'compression_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Create cross-scene comparison
    create_cross_scene_comparison(all_results)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60)


def create_cross_scene_comparison(all_results):
    """Create comparison across all scenes"""
    if len(all_results) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenes = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenes)))
    
    # 1. Compression Ratio by Scene
    ax1 = axes[0, 0]
    methods = [r['name'] for r in all_results[scenes[0]]]
    x = np.arange(len(methods))
    width = 0.2
    
    for i, scene in enumerate(scenes):
        ratios = [r['ratio'] for r in all_results[scene]]
        ax1.bar(x + i*width, ratios, width, label=scene.capitalize(), color=colors[i])
    
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('Compression Ratio by Scene')
    ax1.set_xticks(x + width * (len(scenes)-1) / 2)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. PSNR by Scene
    ax2 = axes[0, 1]
    for i, scene in enumerate(scenes):
        psnrs = [r['psnr_mean'] for r in all_results[scene]]
        ax2.plot(methods, psnrs, 'o-', label=scene.capitalize(), color=colors[i], linewidth=2, markersize=8)
    
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('PSNR Comparison Across Scenes')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. SSIM by Scene
    ax3 = axes[1, 0]
    for i, scene in enumerate(scenes):
        ssims = [r['ssim_mean'] for r in all_results[scene]]
        ax3.plot(methods, ssims, 's-', label=scene.capitalize(), color=colors[i], linewidth=2, markersize=8)
    
    ax3.set_ylabel('SSIM')
    ax3.set_title('SSIM Comparison Across Scenes')
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for scene in scenes:
        baseline = all_results[scene][0]
        best_compression = max(all_results[scene], key=lambda x: x['ratio'])
        table_data.append([
            scene.capitalize(),
            f"{baseline['psnr_mean']:.2f} dB",
            f"{best_compression['ratio']:.2f}×",
            f"{best_compression['psnr_mean']:.2f} dB"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Scene', 'Baseline PSNR', 'Best Compression', 'Compressed PSNR'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Across Scenes', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output/high_score/cross_scene_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ Cross-scene comparison saved")


if __name__ == "__main__":
    main()
