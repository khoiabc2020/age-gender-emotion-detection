"""
Advanced Data Preprocessing - Ultimate Edition
Tuần 1: Advanced Data Processing
- Gộp datasets (UTKFace, FER2013, FairFace)
- Gộp class Disgust vào Angry
- Lọc nhiễu (kích thước, không chính diện)
- CLAHE preprocessing
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict


def detect_face_quality(image_path, min_size=48):
    """
    Kiểm tra chất lượng khuôn mặt
    - Kích thước tối thiểu
    - Có thể mở rộng: phát hiện khuôn mặt chính diện
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Kiểm tra kích thước tối thiểu
        if width < min_size or height < min_size:
            return False, "Too small"
        
        # Kiểm tra tỷ lệ khung hình hợp lý (không quá dẹt hoặc quá dài)
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Invalid aspect ratio"
        
        # Kiểm tra ảnh có dữ liệu không
        img_array = np.array(img)
        if img_array.size == 0:
            return False, "Empty image"
        
        # Kiểm tra độ tương phản (tránh ảnh quá tối hoặc quá sáng)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Kiểm tra độ tương phản
        std_dev = np.std(gray)
        if std_dev < 10:  # Ảnh quá đơn điệu (có thể là ảnh đen hoặc trắng)
            return False, "Low contrast"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error: {str(e)}"


def apply_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Khắc phục lỗi ánh sáng yếu
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Save
        cv2.imwrite(str(output_path), img_clahe)
        return True
    except Exception as e:
        print(f"Error applying CLAHE to {image_path}: {e}")
        return False


def merge_datasets(
    utkface_dir=None,
    fer2013_dir=None,
    fairface_dir=None,
    output_dir=None,
    merge_disgust_to_angry=True,
    apply_clahe_preprocessing=True,
    min_face_size=48
):
    """
    Gộp nhiều datasets thành một dataset thống nhất
    
    Args:
        utkface_dir: Path to processed UTKFace dataset
        fer2013_dir: Path to processed FER2013 dataset
        fairface_dir: Path to FairFace dataset (optional)
        output_dir: Output directory for merged dataset
        merge_disgust_to_angry: Gộp class Disgust vào Angry
        apply_clahe_preprocessing: Áp dụng CLAHE preprocessing
        min_face_size: Kích thước khuôn mặt tối thiểu
    """
    print("=" * 60)
    print("MERGING DATASETS - Advanced Preprocessing")
    print("=" * 60)
    
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / "merged"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Emotion mapping (gộp Disgust vào Angry)
    emotion_map = {
        'angry': 0,
        'disgust': 0,  # Gộp vào Angry
        'fear': 2,
        'happy': 4,
        'neutral': 3,
        'sad': 5,
        'surprise': 6
    }
    
    # Emotion names (6 classes sau khi gộp)
    emotion_names = ['angry', 'fear', 'neutral', 'happy', 'sad', 'surprise']
    
    stats = defaultdict(int)
    
    # Process UTKFace
    if utkface_dir and Path(utkface_dir).exists():
        print("\n📁 Processing UTKFace dataset...")
        utkface_path = Path(utkface_dir)
        
        for split in ['train', 'val', 'test']:
            split_dir = utkface_path / split
            if not split_dir.exists():
                continue
            
            output_split_dir = train_dir if split == 'train' else (val_dir if split == 'val' else test_dir)
            
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            
            for img_path in tqdm(images, desc=f"UTKFace {split}"):
                # Kiểm tra chất lượng
                is_valid, reason = detect_face_quality(img_path, min_size=min_face_size)
                if not is_valid:
                    stats[f'utkface_{split}_filtered'] += 1
                    continue
                
                # Parse labels từ tên file
                filename = img_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    try:
                        age = int(parts[0])
                        gender = int(parts[1])
                        
                        # Validate
                        if age < 0 or age > 100:
                            continue
                        if gender not in [0, 1]:
                            gender = 0
                        
                        # Default emotion: neutral
                        emotion = 3
                        
                        # Save với format: age_gender_emotion_xxx.jpg
                        output_filename = f"{age}_{gender}_{emotion}_{img_path.name}"
                        output_path = output_split_dir / output_filename
                        
                        # Apply CLAHE nếu cần
                        if apply_clahe_preprocessing:
                            apply_clahe(img_path, output_path)
                        else:
                            shutil.copy2(img_path, output_path)
                        
                        stats[f'utkface_{split}_processed'] += 1
                    except (ValueError, IndexError):
                        continue
    
    # Process FER2013
    if fer2013_dir and Path(fer2013_dir).exists():
        print("\n📁 Processing FER2013 dataset...")
        fer2013_path = Path(fer2013_dir)
        
        for split in ['train', 'val', 'test']:
            split_dir = fer2013_path / split
            if not split_dir.exists():
                continue
            
            output_split_dir = train_dir if split == 'train' else (val_dir if split == 'val' else test_dir)
            
            # Process từng emotion class
            for emotion_dir in split_dir.iterdir():
                if not emotion_dir.is_dir():
                    continue
                
                emotion_name = emotion_dir.name.lower()
                
                # Map emotion (gộp Disgust vào Angry)
                if merge_disgust_to_angry and emotion_name == 'disgust':
                    emotion_name = 'angry'
                
                if emotion_name not in emotion_map:
                    continue
                
                emotion_id = emotion_map[emotion_name]
                
                images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
                
                for img_path in tqdm(images, desc=f"FER2013 {split}/{emotion_name}"):
                    # Kiểm tra chất lượng
                    is_valid, reason = detect_face_quality(img_path, min_size=min_face_size)
                    if not is_valid:
                        stats[f'fer2013_{split}_filtered'] += 1
                        continue
                    
                    # FER2013 không có age/gender, dùng default
                    age = 30  # Default age
                    gender = 0  # Default gender
                    
                    # Save với format: age_gender_emotion_xxx.jpg
                    output_filename = f"{age}_{gender}_{emotion_id}_{img_path.name}"
                    output_path = output_split_dir / output_filename
                    
                    # Apply CLAHE nếu cần
                    if apply_clahe_preprocessing:
                        apply_clahe(img_path, output_path)
                    else:
                        shutil.copy2(img_path, output_path)
                    
                    stats[f'fer2013_{split}_processed'] += 1
    
    # Process FairFace (nếu có)
    if fairface_dir and Path(fairface_dir).exists():
        print("\n📁 Processing FairFace dataset...")
        # TODO: Implement FairFace processing
        print("⚠️  FairFace processing not yet implemented")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING STATISTICS")
    print("=" * 60)
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Merged dataset saved to: {output_dir}")
    print("=" * 60)
    
    return output_dir


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge and preprocess datasets')
    parser.add_argument('--utkface_dir', type=str, default=None,
                        help='Path to processed UTKFace dataset')
    parser.add_argument('--fer2013_dir', type=str, default=None,
                        help='Path to processed FER2013 dataset')
    parser.add_argument('--fairface_dir', type=str, default=None,
                        help='Path to FairFace dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for merged dataset')
    parser.add_argument('--no_clahe', action='store_true',
                        help='Disable CLAHE preprocessing')
    parser.add_argument('--no_merge_disgust', action='store_true',
                        help='Do not merge Disgust into Angry')
    parser.add_argument('--min_face_size', type=int, default=48,
                        help='Minimum face size in pixels')
    
    args = parser.parse_args()
    
    # Auto-detect paths if not provided
    project_root = Path(__file__).parent.parent.parent.parent
    
    if args.utkface_dir is None:
        utkface_path = project_root / "data" / "processed" / "utkface"
        if utkface_path.exists():
            args.utkface_dir = str(utkface_path)
    
    if args.fer2013_dir is None:
        fer2013_path = project_root / "data" / "processed" / "fer2013"
        if fer2013_path.exists():
            args.fer2013_dir = str(fer2013_path)
    
    merge_datasets(
        utkface_dir=args.utkface_dir,
        fer2013_dir=args.fer2013_dir,
        fairface_dir=args.fairface_dir,
        output_dir=args.output_dir,
        merge_disgust_to_angry=not args.no_merge_disgust,
        apply_clahe_preprocessing=not args.no_clahe,
        min_face_size=args.min_face_size
    )


if __name__ == "__main__":
    main()

