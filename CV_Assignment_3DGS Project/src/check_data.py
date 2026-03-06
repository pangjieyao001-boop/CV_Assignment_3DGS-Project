#!/usr/bin/env python3
"""
数据集验证脚本
检查下载的数据集是否完整可用
"""

import sys
from pathlib import Path
from utils.data_loader import SyntheticDataLoader


def check_dataset(data_path='data/lego'):
    """检查数据集完整性"""
    data_path = Path(data_path)
    
    print("="*60)
    print("Dataset Validation Check")
    print("="*60)
    print(f"Checking: {data_path.absolute()}")
    
    # 检查目录是否存在
    if not data_path.exists():
        print(f"\n❌ Directory not found: {data_path}")
        print("\nPlease download the dataset first:")
        print("  nerfbaselines download-dataset external://blender/lego -o data/lego")
        return False
    
    # 检查关键文件
    required_files = ['transforms_train.json', 'transforms_test.json']
    images_dir = data_path / 'images'
    
    for file in required_files:
        file_path = data_path / file
        if file_path.exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
            return False
    
    # 检查图像目录（支持多种格式）
    image_dirs = [images_dir, data_path / 'train', data_path / 'test']
    found_images = False
    for img_dir in image_dirs:
        if img_dir.exists():
            image_count = len(list(img_dir.glob('*.png')))
            if image_count > 0:
                print(f"✅ {img_dir.name}/ directory found ({image_count} images)")
                found_images = True
                break
    
    if not found_images:
        print(f"❌ No image directories found")
        return False
    
    # 尝试加载数据
    print("\n" + "-"*60)
    print("Loading data...")
    try:
        loader = SyntheticDataLoader(data_path)
        
        # 加载训练集
        train_cameras = loader.load_cameras('train')
        print(f"✅ Train cameras: {len(train_cameras)}")
        
        # 加载测试集
        test_cameras = loader.load_cameras('test')
        print(f"✅ Test cameras: {len(test_cameras)}")
        
        if len(train_cameras) > 0:
            # 检查图像格式
            img = train_cameras[0]['image']
            print(f"✅ Image shape: {img.shape}")
            print(f"✅ Image dtype: {img.dtype}")
            
            # 检查相机参数
            print(f"✅ FoVx: {train_cameras[0]['FoVx']:.4f}")
            print(f"✅ FoVy: {train_cameras[0]['FoVy']:.4f}")
        
        print("\n" + "="*60)
        print("✅ Dataset validation PASSED!")
        print("="*60)
        print("\nYou can now run real experiments:")
        print("  python3 run_experiments.py -s data/lego -m output/real --full --iterations 7000")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate downloaded dataset')
    parser.add_argument('--path', type=str, default='data/lego',
                       help='Path to dataset (default: data/lego)')
    
    args = parser.parse_args()
    
    success = check_dataset(args.path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
