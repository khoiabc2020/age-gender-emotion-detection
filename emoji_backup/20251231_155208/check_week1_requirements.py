"""
Kiểm tra Tuần 1: Chuẩn bị & Xử lý dữ liệu
- Dataset: UTKFace, FER2013
- Data Cleaning: Gộp Disgust -> Angry
- Data Augmentation: Albumentations advanced
"""

import sys
import io
from pathlib import Path
import importlib.util

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "training_experiments" / "src"))

def check_datasets():
    """Kiểm tra datasets có tồn tại không"""
    print("=" * 60)
    print("📊 KIỂM TRA DATASETS")
    print("=" * 60)
    
    data_dir = project_root / "training_experiments" / "data"
    results = []
    
    # Check UTKFace
    print("\n[1/2] Checking UTKFace dataset...")
    utkface_paths = [
        data_dir / "utkface" / "UTKFace",
        data_dir / "utkface" / "crop_part1",
        data_dir / "processed" / "utkface"
    ]
    
    utkface_found = False
    for path in utkface_paths:
        if path.exists():
            images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if len(images) > 0:
                print(f"   ✅ UTKFace found: {path}")
                print(f"      Images: {len(images)}")
                utkface_found = True
                break
    
    if not utkface_found:
        print("   ❌ UTKFace dataset not found")
        print("      Expected locations:")
        for path in utkface_paths:
            print(f"        - {path}")
    results.append(("UTKFace Dataset", utkface_found))
    
    # Check FER2013
    print("\n[2/2] Checking FER2013 dataset...")
    fer2013_paths = [
        data_dir / "fer2013" / "train",
        data_dir / "fer2013" / "test",
        data_dir / "processed" / "fer2013"
    ]
    
    fer2013_found = False
    for path in fer2013_paths:
        if path.exists():
            # Check for emotion directories
            emotion_dirs = [d for d in path.iterdir() if d.is_dir()]
            if len(emotion_dirs) > 0:
                total_images = sum(
                    len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                    for d in emotion_dirs
                )
                if total_images > 0:
                    print(f"   ✅ FER2013 found: {path}")
                    print(f"      Emotion classes: {len(emotion_dirs)}")
                    print(f"      Total images: {total_images}")
                    fer2013_found = True
                    break
    
    if not fer2013_found:
        print("   ❌ FER2013 dataset not found")
        print("      Expected locations:")
        for path in fer2013_paths:
            print(f"        - {path}")
    results.append(("FER2013 Dataset", fer2013_found))
    
    return results


def check_disgust_merge():
    """Kiểm tra logic gộp Disgust -> Angry"""
    print("\n" + "=" * 60)
    print("🧹 KIỂM TRA DATA CLEANING (Gộp Disgust -> Angry)")
    print("=" * 60)
    
    results = []
    
    # Check advanced_preprocess.py
    print("\n[1/2] Checking advanced_preprocess.py...")
    preprocess_file = project_root / "training_experiments" / "src" / "data" / "advanced_preprocess.py"
    
    if preprocess_file.exists():
        with open(preprocess_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        has_merge = 'merge_disgust_to_angry' in content.lower()
        has_emotion_map = 'disgust' in content.lower() and 'angry' in content.lower()
        
        if has_merge and has_emotion_map:
            print("   ✅ Disgust -> Angry merge logic found")
            print("      - merge_disgust_to_angry parameter")
            print("      - emotion_map with disgust -> angry")
            results.append(("advanced_preprocess.py", True))
        else:
            print("   ⚠️  Disgust merge logic may be incomplete")
            results.append(("advanced_preprocess.py", False))
    else:
        print("   ❌ advanced_preprocess.py not found")
        results.append(("advanced_preprocess.py", False))
    
    # Check dataset.py
    print("\n[2/2] Checking dataset.py...")
    dataset_file = project_root / "training_experiments" / "src" / "data" / "dataset.py"
    
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        has_emotion_map = 'disgust' in content.lower() and 'angry' in content.lower()
        has_6_classes = 'num_emotions=6' in content or '6 classes' in content.lower()
        
        if has_emotion_map:
            print("   ✅ Disgust -> Angry mapping in dataset.py")
            if has_6_classes:
                print("      - 6 emotion classes (Disgust merged)")
            results.append(("dataset.py", True))
        else:
            print("   ⚠️  Disgust mapping may be missing")
            results.append(("dataset.py", False))
    else:
        print("   ❌ dataset.py not found")
        results.append(("dataset.py", False))
    
    return results


def check_albumentations():
    """Kiểm tra Albumentations augmentations"""
    print("\n" + "=" * 60)
    print("🎨 KIỂM TRA DATA AUGMENTATION (Albumentations)")
    print("=" * 60)
    
    results = []
    
    # Check if albumentations is installed
    print("\n[1/3] Checking Albumentations package...")
    try:
        import albumentations as A
        print(f"   ✅ Albumentations installed: {A.__version__}")
        results.append(("Albumentations Package", True))
    except ImportError:
        print("   ❌ Albumentations not installed")
        print("      Install: pip install albumentations")
        results.append(("Albumentations Package", False))
        return results
    
    # Check dataset.py for augmentations
    print("\n[2/3] Checking augmentation transforms in dataset.py...")
    dataset_file = project_root / "training_experiments" / "src" / "data" / "dataset.py"
    
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for advanced augmentations
        augmentations = {
            'HorizontalFlip': 'A.HorizontalFlip' in content,
            'Rotate': 'A.Rotate' in content,
            'ShiftScaleRotate': 'A.ShiftScaleRotate' in content,
            'Perspective': 'A.Perspective' in content,
            'RandomBrightnessContrast': 'A.RandomBrightnessContrast' in content,
            'HueSaturationValue': 'A.HueSaturationValue' in content,
            'CLAHE': 'A.CLAHE' in content,
            'RandomGamma': 'A.RandomGamma' in content,
            'GaussNoise': 'A.GaussNoise' in content,
            'MotionBlur': 'A.MotionBlur' in content,
            'GaussianBlur': 'A.GaussianBlur' in content,
            'CoarseDropout': 'A.CoarseDropout' in content,
            'GridDistortion': 'A.GridDistortion' in content,
            'GridDropout': 'A.GridDropout' in content or 'GridMask' in content,
        }
        
        found_count = sum(augmentations.values())
        total_count = len(augmentations)
        
        print(f"   Found {found_count}/{total_count} augmentations:")
        for aug_name, found in augmentations.items():
            status = "✅" if found else "❌"
            print(f"      {status} {aug_name}")
        
        if found_count >= 10:
            print(f"\n   ✅ Advanced augmentations implemented ({found_count}/{total_count})")
            results.append(("Augmentation Transforms", True))
        else:
            print(f"\n   ⚠️  Some augmentations missing ({found_count}/{total_count})")
            results.append(("Augmentation Transforms", found_count >= 8))
    else:
        print("   ❌ dataset.py not found")
        results.append(("Augmentation Transforms", False))
    
    # Check for MixUp and CutMix
    print("\n[3/3] Checking MixUp & CutMix...")
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_mixup = 'mixup' in content.lower() and '_mixup' in content.lower()
        has_cutmix = 'cutmix' in content.lower() and '_cutmix' in content.lower()
        
        if has_mixup:
            print("   ✅ MixUp augmentation found")
        else:
            print("   ❌ MixUp not found")
        
        if has_cutmix:
            print("   ✅ CutMix augmentation found")
        else:
            print("   ❌ CutMix not found")
        
        results.append(("MixUp & CutMix", has_mixup and has_cutmix))
    
    return results


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TUẦN 1: CHUẨN BỊ & XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    
    all_results = []
    
    # Check datasets
    dataset_results = check_datasets()
    all_results.extend(dataset_results)
    
    # Check Disgust merge
    merge_results = check_disgust_merge()
    all_results.extend(merge_results)
    
    # Check augmentations
    aug_results = check_albumentations()
    all_results.extend(aug_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    print(f"\nKết quả: {passed}/{total} checks passed\n")
    
    for name, result in all_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:40s} {status}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("🎉 Tất cả yêu cầu Tuần 1 đã hoàn thành!")
        print("\nCó thể tiếp tục với Tuần 2: Model Training")
    else:
        print("⚠️  Một số yêu cầu chưa hoàn thành")
        print("\nCần kiểm tra và sửa các phần còn thiếu trước khi tiếp tục")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

