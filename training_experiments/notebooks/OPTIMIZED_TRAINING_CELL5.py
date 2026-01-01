# OPTIMIZED TRAINING CELL - COPY VÀO KAGGLE CELL 5

# ============================================================
# IMPROVED VERSION WITH:
# - Vision Transformer option
# - Mixed precision training (faster)
# - Better augmentation
# - OneCycleLR scheduler
# - Gradient accumulation
# - Enhanced Mixup
# Target: 78-82% with 4 datasets
# ============================================================

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import timm
import json
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

print("=" * 60)
print("OPTIMIZED TRAINING - TARGET 78-82%")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================

# MODEL SELECTION: Change this to 'vit' for Vision Transformer
MODEL_TYPE = 'efficientnet'  # 'efficientnet' or 'vit'

EPOCHS = 150
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2  # Effective batch size = 128
LEARNING_RATE = 0.0003
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 40
USE_MIXED_PRECISION = True  # Faster training

print(f"\n[CONFIG] Optimized Configuration:")
print(f"  Model: {MODEL_TYPE.upper()}")
print(f"  Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load dataset paths
with open('/kaggle/working/dataset_paths.json') as f:
    dataset_paths = json.load(f)

print(f"\n[INFO] Found {len(dataset_paths)} datasets:")
for name in dataset_paths.keys():
    print(f"  - {name.upper()}")

# ============================================================
# IMPROVED DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.3)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("[INFO] Enhanced data augmentation applied")

# ============================================================
# LOAD DATASETS (same as before)
# ============================================================

print("\n[INFO] Loading datasets...")
all_train_datasets, all_test_datasets = [], []
total_train, total_test = 0, 0

for name, path in dataset_paths.items():
    dataset_path = Path(path)
    train_dir = test_dir = None
    
    if (dataset_path / 'train').exists():
        train_dir = dataset_path / 'train'
        test_dir = dataset_path / 'test' if (dataset_path / 'test').exists() else train_dir
    else:
        train_dirs = list(dataset_path.glob('**/train'))
        if train_dirs:
            train_dir = train_dirs[0]
            test_dir = train_dir.parent / 'test'
            if not test_dir.exists():
                test_dir = train_dir.parent / 'validation' if (train_dir.parent / 'validation').exists() else train_dir
    
    if train_dir and train_dir.exists():
        try:
            train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
            test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
            if len(train_ds) > 0:
                all_train_datasets.append(train_ds)
                all_test_datasets.append(test_ds)
                total_train += len(train_ds)
                total_test += len(test_ds)
                print(f"  [OK] {name.upper()}: {len(train_ds):,} train, {len(test_ds):,} test")
        except Exception as e:
            print(f"  [WARN] {name.upper()}: {str(e)[:50]}")

train_dataset = ConcatDataset(all_train_datasets) if len(all_train_datasets) > 1 else all_train_datasets[0]
test_dataset = ConcatDataset(all_test_datasets) if len(all_test_datasets) > 1 else all_test_datasets[0]

print(f"\n[SUCCESS] Total: {total_train:,} train, {total_test:,} test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

first_dataset = all_train_datasets[0]
num_classes = len(first_dataset.classes)
class_names = first_dataset.classes

print(f"[OK] Classes: {num_classes}, Batches: {len(train_loader)} train")

# ============================================================
# MODEL: EFFICIENTNET OR VISION TRANSFORMER
# ============================================================

print(f"\n[INFO] Creating {MODEL_TYPE.upper()} model...")

if MODEL_TYPE == 'vit':
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes, drop_rate=0.3)
    print("[INFO] Vision Transformer (may achieve 80-85% with 50K+ images)")
else:
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes, drop_rate=0.5)

model = model.to(DEVICE)
print(f"[OK] Model ready ({sum(p.numel() for p in model.parameters()):,} parameters)")

# ============================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3.0, label_smoothing=0.15):
        super().__init__()
        self.alpha, self.gamma, self.label_smoothing = alpha, gamma, label_smoothing
    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(inputs, targets)
        return self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce

criterion = ImprovedFocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS,
    steps_per_epoch=len(train_loader)//GRAD_ACCUM_STEPS, pct_start=0.1
)
scaler = GradScaler() if USE_MIXED_PRECISION else None

print("[OK] ImprovedFocalLoss + AdamW + OneCycleLR + MixedPrecision")

# ============================================================
# MIXUP
# ============================================================

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    lam = max(lam, 1-lam)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def mixup_loss(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1-lam) * crit(pred, y_b)

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "="*60)
print(f"STARTING TRAINING - {total_train:,} images")
print("="*60)

best_accuracy, best_epoch, patience = 0, 0, 0
start_time = time.time()
history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}

for epoch in range(EPOCHS):
    model.train()
    train_loss = train_correct = train_total = 0
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'E{epoch+1}')):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        if USE_MIXED_PRECISION:
            with autocast():
                if np.random.rand() > 0.3:
                    images, la, lb, lam = mixup_data(images, labels)
                    loss = mixup_loss(criterion, model(images), la, lb, lam) / GRAD_ACCUM_STEPS
                else:
                    loss = criterion(model(images), labels) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            if np.random.rand() > 0.3:
                images, la, lb, lam = mixup_data(images, labels)
                loss = mixup_loss(criterion, model(images), la, lb, lam) / GRAD_ACCUM_STEPS
            else:
                loss = criterion(model(images), labels) / GRAD_ACCUM_STEPS
            loss.backward()
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        with torch.no_grad():
            train_loss += loss.item() * GRAD_ACCUM_STEPS
            _, pred = model(images).max(1) if USE_MIXED_PRECISION else (None, model(images).max(1)[1])
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Validation
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            test_correct += outputs.max(1)[1].eq(labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100. * test_correct / test_total
    history['train_loss'].append(float(train_loss/len(train_loader)))
    history['train_acc'].append(float(train_acc))
    history['test_acc'].append(float(test_acc))
    history['lr'].append(float(optimizer.param_groups[0]['lr']))
    
    print(f"\nE{epoch+1}: Train={train_acc:.2f}%, Val={test_acc:.2f}%, Time={((time.time()-start_time)/60):.1f}m")
    
    if test_acc > best_accuracy:
        best_accuracy, best_epoch, patience = test_acc, epoch+1, 0
        save_dir = Path('/kaggle/working/checkpoints_production')
        save_dir.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_accuracy': best_accuracy,
            'class_names': class_names,
            'num_classes': num_classes,
            'model_type': MODEL_TYPE
        }, save_dir / 'best_model_4datasets.pth')
        print(f"[BEST] {best_accuracy:.2f}%")
    else:
        patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"\n[STOP] Early stopping")
            break

# Save results
with open('/kaggle/working/checkpoints_production/training_results.json', 'w') as f:
    json.dump({
        'best_accuracy': float(best_accuracy),
        'best_epoch': best_epoch,
        'total_epochs': epoch+1,
        'training_time_hours': (time.time()-start_time)/3600,
        'total_train_images': total_train,
        'model_type': MODEL_TYPE,
        'history': history
    }, f, indent=2)

print("\n" + "="*60)
print(f"COMPLETE: {best_accuracy:.2f}% in {((time.time()-start_time)/3600):.2f}h")
print("="*60)
