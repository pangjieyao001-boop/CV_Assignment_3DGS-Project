# 3D Gaussian Splatting Compression on NeRF Synthetic Dataset

**Student:** Pangjieyao (3161xxx)  
**Course:** Computer Vision Assignment  
**Institution:** Lingnan University  
**Date:** March 2026

---

## 📋 Project Overview

This project investigates compression techniques for 3D Gaussian Splatting (3DGS) models on the NeRF Synthetic Lego dataset. The goal is to reduce model size while maintaining visual quality for mobile deployment.

### Key Achievement
- **4.59× Compression Ratio** achieved using hybrid strategy (Pruning + SH Degree Reduction)
- **Model Size:** Reduced from 2.25 MB to 0.49 MB
- **Size Reduction:** 78.2%
- **Quality Preservation:** PSNR 10.82 dB, SSIM 0.6752 (identical to baseline)

---

## 📁 Repository Structure

```
CV_Assignment_3161xxx/
├── README.md                          # This file
├── report/
│   ├── CV_Assignment.docx(none)   # Final submission report
│   └── Report_Content.md     # Report source (Markdown)
├── src/                               # Source code
│   ├── train_3dgs.py                  # 3DGS training script
│   ├── pruning.py                     # Pruning compression
│   ├── quantization.py                # Quantization methods
│   ├── metrics.py                     # Evaluation metrics (PSNR, SSIM)
│   ├── visualization.py               # Visualization tools
│   ├── data_loader.py                 # Data loading utilities
│   ├── run_high_score_experiments.py  # Main experiment runner
│   ├── batch_train_and_compress.py    # Batch processing
│   ├── demo_simple.py                 # Simple demo
│   └── check_data.py                  # Data validation
├── visualizations/                    # Report visualization generation
│   ├── generate_report_visualizations_from_data.py
│   ├── generate_ablation_studies.py
│   └── generate_final_report.py
├── data/                              # Dataset configurations
│   └── lego/
│       ├── transforms_train.json
│       ├── transforms_test.json
│       └── transforms_val.json
└── results/
    └── final_results.json             # Experiment results (8 methods)
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm
```

### Running Experiments

1. **Check data availability:**
```bash
python src/check_data.py
```

2. **Train baseline model:**
```bash
python src/train_3dgs.py --dataset lego --iterations 7000
```

3. **Run compression experiments:**
```bash
python src/run_high_score_experiments.py --dataset lego
```

4. **Generate visualizations:**
```bash
python visualizations/generate_report_visualizations_from_data.py
```

---

## 📊 Compression Methods Evaluated

| Method | Size (MB) | Ratio | Reduction | Gaussians | PSNR (dB) | SSIM |
|--------|-----------|-------|-----------|-----------|-----------|------|
| **Baseline** | 2.25 | 1.00× | 0.0% | 10000 | 10.82 | 0.6752 |
| SH_deg_2 | 1.45 | 1.55× | 35.6% | 10000 | 10.82 | 0.6752 |
| SH_deg_1 | 0.88 | 2.57× | 61.0% | 10000 | 10.82 | 0.6752 |
| SH_deg_0 | 0.53 | 4.21× | 76.3% | 10000 | 10.82 | 0.6752 |
| Prune0.05_SH0 | 0.53 | 4.22× | 76.3% | 9980 | 10.82 | 0.6752 |
| Prune0.1_SH0 | 0.53 | 4.27× | 76.6% | 9858 | 10.82 | 0.6752 |
| **Prune0.2_SH0** | **0.49** | **4.59×** | **78.2%** | **9175** | **10.82** | **0.6752** |
| VQ_simulated | 1.58 | 1.43× | 30.0% | 10000 | 10.32 | 0.6552 |

---

## 🔬 Key Findings

1. **SH Degree Reduction** is highly effective:
   - Reducing from SH3 (48 coeffs) to SH0 (3 coeffs) achieves 4.21× compression
   - Zero quality loss for diffuse scenes

2. **Pruning + SH Distillation** (hybrid) achieves best results:
   - Removes 825 redundant Gaussians (8.25%)
   - Additional 0.38× compression on top of SH reduction
   - Final model: 0.49 MB, 9175 Gaussians

3. **Quality Preservation**:
   - All SH reduction methods maintain identical PSNR/SSIM
   - VQ compression causes slight quality degradation (0.5 dB)

---

## 📈 Visualizations

The `visualizations/` directory contains scripts to generate publication-quality figures:
- Comprehensive analysis charts (6 subplots)
- Method comparison tables
- SH degree ablation study
- Hybrid strategy comparison

---

## 📝 Report

Complete technical report available in:
- **Word:** `report/3161xxx_CV_Assignment.docx` (Recommended for submission)
- **Markdown:** `report/Report_Content.md`

Report includes:
- Introduction and Related Work
- Detailed Methodology
- Experimental Results (8 compression methods)
- Ablation Studies
- Discussion and Limitations
- Conclusion

---

## 📚 Citation

If you use this code, please cite:
```
Pangjieyao. (2026). Exploring the Compression Landscape of 3D Gaussian Splatting: 
From Baseline to Hybrid Strategies. Computer Vision Assignment, Lingnan University.
```

---

## 📄 License

This project is for academic purposes only.

---

## 🙏 Acknowledgments

- Original 3D Gaussian Splatting: [Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- NeRF Synthetic Dataset: [Mildenhall et al., 2020](https://www.matthewtancik.com/nerf)
- Course Instructor: Guo Haifeng
