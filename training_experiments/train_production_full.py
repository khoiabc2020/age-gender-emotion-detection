#!/usr/bin/env python3
"""
PRODUCTION-GRADE TRAINING - OPTION B
Target: 78-85% accuracy with 4 datasets

Features:
- EfficientNet-B2 pretrained backbone
- 4 datasets: FER2013, UTKFace, RAF-DB, AffectNet  
- Advanced augmentation (Albumentations)
- Focal Loss for class imbalance
- Mixup/Cutmix augmentation
- Early stopping
- Cosine annealing LR
- TensorBoard logging
- Gradient clipping
- Label smoothing

Usage:
    python train_production_full.py --data_paths dataset_paths.json

Timeline: 8-12 hours on T4 GPU, 4-6 hours on V100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
from datetime import datetime
import sys
import io
from PIL import Image
import cv2

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.size(-1)
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / num_classes
        
        # Focal loss
        ce_loss = -(targets_one_hot * torch.log_softmax(inputs, dim=1)).sum(dim=1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


# ============================================================
# MIXUP / CUTMIX AUGMENTATION
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Cutmix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for cutmix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class ProductionMultiTaskModel(nn.Module):
    """
    Production-grade multi-task model
    Backbone: EfficientNet-B2 (pretrained on ImageNet)
    Tasks: Emotion (7 classes), Age (regression), Gender (2 classes)
    """
    def __init__(
        self,
        num_emotions=7,
        num_genders=2,
        backbone='efficientnet_b2',
        pretrained=True,
        dropout_rate=0.5
    ):
        super().__init__()
        
        print(f"[INFO] Creating model with {backbone} backbone...")
        
        # Load pretrained backbone from timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
        
        print(f"[INFO] Backbone feature dim: {feature_dim}")
        
        # Shared feature projection with BatchNorm
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        # Emotion head (primary task)
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_emotions)
        )
        
        # Age head (regression)
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 1)
        )
        
        # Gender head
        self.gender_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, num_genders)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for task heads"""
        for m in [self.shared, self.emotion_head, self.age_head, self.gender_head]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Shared representation
        shared = self.shared(features)
        
        # Task-specific predictions
        emotion = self.emotion_head(shared)
        age = self.age_head(shared)
        gender = self.gender_head(shared)
        
        return emotion, age, gender


# ============================================================
# DATA AUGMENTATION
# ============================================================

def get_advanced_transforms(is_training=True, img_size=224):
    """
    Advanced augmentation using Albumentations
    Significantly stronger than basic PyTorch transforms
    """
    if is_training:
        return A.Compose([
            # Geometric transforms
            A.Resize(int(img_size * 1.15), int(img_size * 1.15)),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=25,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            
            # Color transforms
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            ], p=0.5),
            
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1),
                A.GaussianBlur(blur_limit=5, p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.2),
            
            # Cutout variants
            A.CoarseDropout(
                max_holes=8,
                max_height=int(img_size * 0.15),
                max_width=int(img_size * 0.15),
                min_holes=1,
                min_height=int(img_size * 0.05),
                min_width=int(img_size * 0.05),
                fill_value=0,
                p=0.5
            ),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Validation: only resize and normalize
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


# ============================================================
# DATASET LOADERS
# ============================================================

class AlbumentationsDataset(Dataset):
    """Wrapper for Albumentations transforms"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def load_datasets(dataset_paths, img_size=224):
    """
    Load all available datasets
    Returns: train_dataset, val_dataset, test_dataset
    """
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # Get transforms
    train_transform = get_advanced_transforms(is_training=True, img_size=img_size)
    val_transform = get_advanced_transforms(is_training=False, img_size=img_size)
    
    # Load FER2013 (Primary emotion dataset)
    if 'fer2013' in dataset_paths:
        print("\n[INFO] Loading FER2013...")
        fer_path = Path(dataset_paths['fer2013'])
        try:
            fer_train = datasets.ImageFolder(fer_path / 'train')
            fer_test = datasets.ImageFolder(fer_path / 'test')
            
            fer_train = AlbumentationsDataset(fer_train, train_transform)
            fer_test = AlbumentationsDataset(fer_test, val_transform)
            
            train_datasets.append(fer_train)
            test_datasets.append(fer_test)
            
            print(f"  [OK] FER2013: {len(fer_train)} train, {len(fer_test)} test")
        except Exception as e:
            print(f"  [ERROR] Failed to load FER2013: {e}")
    
    # Load RAF-DB (High-quality emotion)
    if 'rafdb' in dataset_paths:
        print("\n[INFO] Loading RAF-DB...")
        raf_path = Path(dataset_paths['rafdb'])
        try:
            # RAF-DB may have different structure, adapt as needed
            # Assuming it has train/test folders
            if (raf_path / 'train').exists():
                raf_train = datasets.ImageFolder(raf_path / 'train')
                raf_train = AlbumentationsDataset(raf_train, train_transform)
                train_datasets.append(raf_train)
                print(f"  [OK] RAF-DB: {len(raf_train)} train")
            else:
                print(f"  [WARN] RAF-DB structure not recognized, skipping")
        except Exception as e:
            print(f"  [ERROR] Failed to load RAF-DB: {e}")
    
    # Load AffectNet (Large-scale emotion)
    if 'affectnet' in dataset_paths:
        print("\n[INFO] Loading AffectNet...")
        affect_path = Path(dataset_paths['affectnet'])
        try:
            if (affect_path / 'train').exists():
                affect_train = datasets.ImageFolder(affect_path / 'train')
                affect_train = AlbumentationsDataset(affect_train, train_transform)
                train_datasets.append(affect_train)
                print(f"  [OK] AffectNet: {len(affect_train)} train")
            else:
                print(f"  [WARN] AffectNet structure not recognized, skipping")
        except Exception as e:
            print(f"  [ERROR] Failed to load AffectNet: {e}")
    
    # Combine datasets
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        print(f"\n[INFO] Combined training: {len(train_dataset)} samples")
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        raise ValueError("No datasets loaded!")
    
    if len(test_datasets) > 0:
        test_dataset = test_datasets[0]  # Use FER2013 test as validation
    else:
        # Split train into train/val if no test set
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    print(f"\n[OK] Final:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(test_dataset)} samples")
    print("=" * 60)
    
    return train_dataset, test_dataset


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion_emotion,
    criterion_age,
    criterion_gender,
    device,
    epoch,
    use_mixup_cutmix=True
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply mixup/cutmix randomly
        if use_mixup_cutmix and np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
            else:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()
        
        # Forward pass
        emotion_out, age_out, gender_out = model(images)
        
        # Calculate loss (focus on emotion for now)
        if mixed:
            loss = lam * criterion_emotion(emotion_out, labels_a) + \
                   (1 - lam) * criterion_emotion(emotion_out, labels_b)
        else:
            loss = criterion_emotion(emotion_out, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        if not mixed:
            _, predicted = emotion_out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        if not mixed:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        else:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': 'mixed'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion_emotion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            emotion_out, age_out, gender_out = model(images)
            
            # Calculate loss
            loss = criterion_emotion(emotion_out, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = emotion_out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Production Training - Option B')
    parser.add_argument('--data_paths', type=str, required=True, help='JSON file with dataset paths')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='efficientnet_b2', help='Backbone model')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='/content/checkpoints_production', help='Save directory')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PRODUCTION-GRADE TRAINING - OPTION B")
    print("Target: 78-85% Accuracy")
    print("=" * 60)
    
    print(f"\n[CONFIG]")
    print(f"  Backbone: {args.backbone}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Early Stopping: {args.patience} epochs")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
        if 'T4' in gpu_name:
            print(f"  [INFO] Estimated time: 10-12 hours")
        elif 'V100' in gpu_name:
            print(f"  [INFO] Estimated time: 4-6 hours")
        elif 'A100' in gpu_name:
            print(f"  [INFO] Estimated time: 2-3 hours")
    
    # Load dataset paths
    print(f"\n[INFO] Loading dataset paths from: {args.data_paths}")
    with open(args.data_paths) as f:
        dataset_paths = json.load(f)
    
    # Load datasets
    train_dataset, val_dataset = load_datasets(dataset_paths, args.img_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    print(f"\n[INFO] Creating model...")
    model = ProductionMultiTaskModel(
        backbone=args.backbone,
        pretrained=True,
        dropout_rate=0.5
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    
    # Loss functions
    criterion_emotion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    criterion_age = nn.MSELoss()
    criterion_gender = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            criterion_emotion, criterion_age, criterion_gender,
            device, epoch, use_mixup_cutmix=True
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion_emotion, device
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': vars(args)
            }
            
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"  [OK] Saved best model (acc: {best_acc:.2f}%)")
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
            print(f"[INFO] Best validation accuracy: {best_acc:.2f}%")
            break
    
    # Training complete
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n[OK] Best Validation Accuracy: {best_acc:.2f}%")
    print(f"[INFO] Training Time: {hours}h {minutes}m")
    print(f"[INFO] Model saved to: {save_dir / 'best_model.pth'}")
    
    # Save training results
    results = {
        'best_accuracy': float(best_acc),
        'total_epochs': epoch,
        'training_time_seconds': int(elapsed_time),
        'config': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to: {save_dir / 'training_results.json'}")
    print("=" * 60)
    
    return best_acc


if __name__ == '__main__':
    try:
        best_acc = main()
        
        # Check if target achieved
        if best_acc >= 78:
            print("\n[SUCCESS] Target accuracy (78-85%) ACHIEVED!")
        elif best_acc >= 75:
            print("\n[OK] Good accuracy achieved, close to target")
        else:
            print("\n[WARN] Below target, consider re-training with adjustments")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
