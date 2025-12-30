"""
Data Preprocessing Script
Tuần 1: Chuẩn bị và tiền xử lý dữ liệu
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def is_valid_image(image_path):
    """
    Kiểm tra ảnh có hợp lệ không
    - Không bị lỗi
    - Không phải ảnh đen trắng (grayscale)
    """
    try:
        img = Image.open(image_path)
        # Kiểm tra mode - phải là RGB
        if img.mode != 'RGB':
            return False
        # Kiểm tra kích thước tối thiểu
        if img.size[0] < 32 or img.size[1] < 32:
            return False
        # Kiểm tra ảnh có dữ liệu không
        img_array = np.array(img)
        if img_array.size == 0:
            return False
        return True
    except (IOError, OSError, ValueError, Exception) as e:
        # Silently fail for invalid images
        return False


def resize_image(image_path, output_path, target_size=(224, 224)):
    """Resize ảnh về kích thước chuẩn"""
    try:
        img = Image.open(image_path)
        # Convert to RGB nếu cần
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize với antialiasing
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_resized.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return False


def split_dataset(source_dir, output_base_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Chia dataset thành train/val/test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Tạo thư mục output
    train_dir = Path(output_base_dir) / "train"
    val_dir = Path(output_base_dir) / "val"
    test_dir = Path(output_base_dir) / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Lấy tất cả ảnh
    source_path = Path(source_dir)
    image_files = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(source_path.rglob(ext))
    
    print(f"Found {len(image_files)} images")
    
    # Lọc ảnh hợp lệ
    valid_images = []
    print("Validating images...")
    for img_path in tqdm(image_files):
        if is_valid_image(img_path):
            valid_images.append(img_path)
    
    print(f"Valid images: {len(valid_images)}")
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(valid_images)
    
    # Chia dataset
    n_total = len(valid_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = valid_images[:n_train]
    val_images = valid_images[n_train:n_train + n_val]
    test_images = valid_images[n_train + n_val:]
    
    print(f"\nSplitting dataset:")
    print(f"  Train: {len(train_images)} ({len(train_images)/n_total*100:.1f}%)")
    print(f"  Val: {len(val_images)} ({len(val_images)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_images)} ({len(test_images)/n_total*100:.1f}%)")
    
    # Copy và resize ảnh
    def process_images(image_list, output_dir, split_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {split_name} set...")
        for img_path in tqdm(image_list):
            # Giữ nguyên tên file
            output_path = output_dir / img_path.name
            # Nếu trùng tên, thêm prefix
            counter = 1
            while output_path.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                output_path = output_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            resize_image(img_path, output_path)
    
    process_images(train_images, train_dir, "Train")
    process_images(val_images, val_dir, "Val")
    process_images(test_images, test_dir, "Test")
    
    print(f"\n[SUCCESS] Dataset split completed!")
    print(f"Output directory: {output_base_dir}")


def preprocess_utkface_dataset():
    """Preprocess UTKFace dataset"""
    print("=" * 60)
    print("Preprocessing UTKFace Dataset")
    print("=" * 60)
    
    # Use dataset from project directory
    project_root = Path(__file__).parent.parent.parent
    utkface_path = project_root / "data" / "utkface"
    
    if not utkface_path.exists():
        print(f"[ERROR] Dataset not found at: {utkface_path}")
        print("Please run: python scripts/copy_datasets_to_project.py")
        return None
    
    print(f"Dataset path: {utkface_path}")
    
    # Tìm thư mục chứa ảnh
    utkface_dir = utkface_path / "UTKFace"
    if not utkface_dir.exists():
        utkface_dir = utkface_path / "crop_part1"
    if not utkface_dir.exists():
        utkface_dir = utkface_path / "utkface_aligned_cropped"
    
    if not utkface_dir.exists():
        print("[ERROR] Could not find UTKFace images directory")
        return None
    
    # Output directory (fix path)
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "processed" / "utkface"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    split_dataset(utkface_dir, output_dir)
    
    return output_dir


def preprocess_fer2013_dataset():
    """Preprocess FER2013 dataset"""
    print("=" * 60)
    print("Preprocessing FER2013 Dataset")
    print("=" * 60)
    
    # Use dataset from project directory
    project_root = Path(__file__).parent.parent.parent
    fer2013_path = project_root / "data" / "fer2013"
    
    if not fer2013_path.exists():
        print(f"[ERROR] Dataset not found at: {fer2013_path}")
        print("Please run: python scripts/copy_datasets_to_project.py")
        return None
    
    print(f"Dataset path: {fer2013_path}")
    
    # FER2013 đã có sẵn train/test split
    source_train = Path(fer2013_path) / "train"
    source_test = Path(fer2013_path) / "test"
    
    if not source_train.exists() or not source_test.exists():
        print("❌ Could not find FER2013 train/test directories")
        return None
    
    # Output directory (fix path)
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "processed" / "fer2013"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy và resize train set
    train_output = output_dir / "train"
    val_output = output_dir / "val"
    test_output = output_dir / "test"
    
    # Chia train thành train/val (80/20)
    print("\nProcessing FER2013 train set...")
    for emotion_dir in tqdm(list(source_train.iterdir())):
        if emotion_dir.is_dir():
            emotion_name = emotion_dir.name
            images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
            
            # Shuffle và chia
            np.random.seed(42)
            np.random.shuffle(images)
            n_train = int(len(images) * 0.8)
            
            train_emotion_dir = train_output / emotion_name
            val_emotion_dir = val_output / emotion_name
            train_emotion_dir.mkdir(parents=True, exist_ok=True)
            val_emotion_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in images[:n_train]:
                output_path = train_emotion_dir / img_path.name
                resize_image(img_path, output_path)
            
            for img_path in images[n_train:]:
                output_path = val_emotion_dir / img_path.name
                resize_image(img_path, output_path)
    
    # Copy test set
    print("\nProcessing FER2013 test set...")
    for emotion_dir in tqdm(list(source_test.iterdir())):
        if emotion_dir.is_dir():
            emotion_name = emotion_dir.name
            test_emotion_dir = test_output / emotion_name
            test_emotion_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
            for img_path in images:
                output_path = test_emotion_dir / img_path.name
                resize_image(img_path, output_path)
    
    print(f"\n[SUCCESS] FER2013 preprocessing completed!")
    print(f"Output directory: {output_dir}")
    
    return output_dir


def main():
    """Main preprocessing function"""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING - Smart Retail Analytics")
    print("=" * 60)
    
    # Preprocess UTKFace
    utkface_output = preprocess_utkface_dataset()
    
    # Preprocess FER2013
    fer2013_output = preprocess_fer2013_dataset()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All preprocessing completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check processed data in data/processed/")
    print("2. Run dataset.py to create DataLoader")
    print("3. Start training!")


if __name__ == "__main__":
    main()

