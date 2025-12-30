"""
Multi-task Dataset for Age, Gender, and Emotion
Optimized với advanced data augmentation và anti-overfitting techniques
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random


class MultiTaskDataset(Dataset):
    """
    Dataset cho Multi-task Learning với advanced augmentation
    Trả về: (image, age, gender, emotion)
    """
    
    def __init__(
        self, 
        data_dir, 
        split='train', 
        image_size=224, 
        use_augmentation=True,
        use_mixup=False,
        use_cutmix=False,
        mixup_alpha=0.2,
        cutmix_alpha=1.0
    ):
        """
        Args:
            data_dir: Thư mục chứa dữ liệu đã preprocess
            split: 'train', 'val', hoặc 'test'
            image_size: Kích thước ảnh đầu vào
            use_augmentation: Có dùng data augmentation không
            use_mixup: Sử dụng MixUp augmentation
            use_cutmix: Sử dụng CutMix augmentation
            mixup_alpha: Alpha parameter cho MixUp
            cutmix_alpha: Alpha parameter cho CutMix
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.use_mixup = use_mixup and (split == 'train')
        self.use_cutmix = use_cutmix and (split == 'train')
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        # Load danh sách ảnh và labels
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"Không tìm thấy ảnh nào trong {self.data_dir / self.split}")
        
        # Setup transforms
        self.transform = self._get_transforms()
    
    def _load_samples(self):
        """Load danh sách ảnh và extract labels từ tên file hoặc cấu trúc thư mục"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Kiểm tra cấu trúc: có thể là flat hoặc theo class
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        if len(subdirs) > 0:
            # Cấu trúc theo class (như FER2013)
            for class_dir in subdirs:
                class_name = class_dir.name
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                
                for img_path in images:
                    # Parse labels từ tên file hoặc class name
                    age, gender, emotion = self._parse_labels(img_path, class_name)
                    samples.append({
                        'image_path': str(img_path),
                        'age': age,
                        'gender': gender,
                        'emotion': emotion
                    })
        else:
            # Cấu trúc flat (như UTKFace với tên file chứa labels)
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            
            for img_path in images:
                age, gender, emotion = self._parse_labels(img_path)
                samples.append({
                    'image_path': str(img_path),
                    'age': age,
                    'gender': gender,
                    'emotion': emotion
                })
        
        return samples
    
    def _parse_labels(self, img_path, class_name=None):
        """
        Parse labels từ tên file hoặc class name
        UTKFace format: [age]_[gender]_[race]_[date&time].jpg
        FER2013: class_name là emotion
        """
        filename = Path(img_path).stem
        
        # Default values
        age = 30  # Default age
        gender = 0  # 0: male, 1: female
        emotion = 3  # 3: neutral (default)
        
        # Parse UTKFace format: age_gender_race_...
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                # Validate gender (0=male, 1=female)
                if gender not in [0, 1]:
                    gender = 0  # Default to male
                # Validate age (0-100)
                if age < 0 or age > 100:
                    age = 30  # Default age
            except (ValueError, IndexError):
                # Invalid format, use defaults
                pass
        
        # Parse emotion từ class name (FER2013)
        # Gộp Disgust vào Angry (6 classes thay vì 7)
        if class_name:
            emotion_map = {
                'angry': 0, 'disgust': 0,  # Gộp Disgust vào Angry
                'fear': 1, 'neutral': 2, 'happy': 3, 'sad': 4, 'surprise': 5
            }
            emotion = emotion_map.get(class_name.lower(), 2)  # Default: neutral
        
        return age, gender, emotion
    
    def _get_transforms(self):
        """Setup advanced data augmentation và transforms"""
        if self.use_augmentation:
            # Advanced augmentation cho training - chống overfitting
            transform = A.Compose([
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    p=0.4
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                
                # Color augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.4
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),  # Fixed: added mean parameter
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                
                # Advanced augmentations
                A.CoarseDropout(
                    max_holes=8,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    p=0.3
                ),  # Fixed: use correct parameter names for latest Albumentations
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
                # GridMask - Advanced augmentation để tránh overfitting
                A.GridDropout(
                    ratio=0.5,
                    holes_number_x=4,
                    holes_number_y=4,
                    p=0.3
                ),  # Fixed: removed deprecated parameters
                
                # Normalize
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Chỉ normalize cho val/test
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return transform
    
    def _mixup(self, image1, image2, label1, label2, alpha=0.2):
        """MixUp augmentation - works with tensors"""
        lam = np.random.beta(alpha, alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Mix labels proportionally
        # For classification: use hard label based on lambda
        # For regression: mix proportionally
        mixed_labels = {
            'age': lam * label1['age'] + (1 - lam) * label2['age'],
            'gender': label1['gender'] if lam > 0.5 else label2['gender'],
            'emotion': label1['emotion'] if lam > 0.5 else label2['emotion']
        }
        
        return mixed_image, mixed_labels, lam
    
    def _cutmix(self, image1, image2, label1, label2, alpha=1.0):
        """CutMix augmentation - works with tensors"""
        lam = np.random.beta(alpha, alpha)
        
        # Get random bounding box
        # image shape: (C, H, W)
        h, w = image1.shape[1], image1.shape[2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Random position
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix (clone để tránh modify original)
        mixed_image = image1.clone()
        mixed_image[:, bby1:bby2, bbx1:bbx2] = image2[:, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        # Mix labels
        mixed_labels = {
            'age': lam * label1['age'] + (1 - lam) * label2['age'],
            'gender': label1['gender'] if lam > 0.5 else label2['gender'],
            'emotion': label1['emotion'] if lam > 0.5 else label2['emotion']
        }
        
        return mixed_image, mixed_labels, lam
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get labels
        age = torch.tensor(sample['age'], dtype=torch.float32)
        gender = torch.tensor(sample['gender'], dtype=torch.long)
        emotion = torch.tensor(sample['emotion'], dtype=torch.long)
        
        label = {
            'age': age,
            'gender': gender,
            'emotion': emotion
        }
        
        # Apply MixUp or CutMix if enabled (only during training)
        if self.use_mixup and random.random() < 0.5:
            idx2 = random.randint(0, len(self.samples) - 1)
            sample2 = self.samples[idx2]
            image2 = Image.open(sample2['image_path']).convert('RGB')
            image2 = np.array(image2)
            if self.transform:
                transformed2 = self.transform(image=image2)
                image2 = transformed2['image']
            else:
                # Convert to tensor if no transform
                image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0
            
            label2 = {
                'age': torch.tensor(sample2['age'], dtype=torch.float32),
                'gender': torch.tensor(sample2['gender'], dtype=torch.long),
                'emotion': torch.tensor(sample2['emotion'], dtype=torch.long)
            }
            
            image, label, _ = self._mixup(image, image2, label, label2, self.mixup_alpha)
        
        elif self.use_cutmix and random.random() < 0.5:
            idx2 = random.randint(0, len(self.samples) - 1)
            sample2 = self.samples[idx2]
            image2 = Image.open(sample2['image_path']).convert('RGB')
            image2 = np.array(image2)
            if self.transform:
                transformed2 = self.transform(image=image2)
                image2 = transformed2['image']
            else:
                image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0
            
            label2 = {
                'age': torch.tensor(sample2['age'], dtype=torch.float32),
                'gender': torch.tensor(sample2['gender'], dtype=torch.long),
                'emotion': torch.tensor(sample2['emotion'], dtype=torch.long)
            }
            
            image, label, _ = self._cutmix(image, image2, label, label2, self.cutmix_alpha)
        
        return {
            'image': image,
            'age': label['age'],
            'gender': label['gender'],
            'emotion': label['emotion']
        }


def get_dataloaders(
    data_dir, 
    batch_size=32, 
    num_workers=4, 
    image_size=224,
    use_mixup=False,
    use_cutmix=False
):
    """
    Tạo DataLoaders cho train, val, test với advanced augmentation
    """
    train_dataset = MultiTaskDataset(
        data_dir, 
        split='train', 
        image_size=image_size, 
        use_augmentation=True,
        use_mixup=use_mixup,
        use_cutmix=use_cutmix
    )
    val_dataset = MultiTaskDataset(
        data_dir, 
        split='val', 
        image_size=image_size, 
        use_augmentation=False
    )
    test_dataset = MultiTaskDataset(
        data_dir, 
        split='test', 
        image_size=image_size, 
        use_augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
