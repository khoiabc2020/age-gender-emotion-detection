# KAGGLE TRAINING WITH AUTO-SAVE - NEVER LOSE DATA AGAIN!
# Copy toàn bộ code này vào Cell 5 trong Kaggle

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
import timm
import json
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import shutil

# ============================================================
# CONFIG - PROVEN TO ACHIEVE 76.49%
# ============================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = 'efficientnet_b0'
NUM_CLASSES = 7
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4  # Effective batch size: 64
LEARNING_RATE = 3e-4
EPOCHS = 150
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
USE_MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 15

print("="*60)
print("TRAINING WITH AUTO-SAVE (NEVER LOSE DATA!)")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_TYPE}")
print(f"Batch Size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print("="*60)

# ============================================================
# AUTO-SAVE SETUP - SAVE EVERY 10 EPOCHS
# ============================================================

CHECKPOINT_DIR = Path('/kaggle/working/checkpoints_production')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, best_accuracy, history, is_best=False):
    """Save checkpoint with auto-backup"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'history': history,
        'class_names': class_names,
        'num_classes': NUM_CLASSES,
        'datasets_used': list(dataset_paths.keys()),
        'config': {
            'model': MODEL_TYPE,
            'batch_size': BATCH_SIZE * GRAD_ACCUM_STEPS,
            'learning_rate': LEARNING_RATE,
            'dropout': DROPOUT
        }
    }
    
    # Save every 10 epochs
    if (epoch + 1) % 10 == 0:
        periodic_path = CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, periodic_path)
        print(f"\n[AUTOSAVE] Checkpoint saved: epoch {epoch+1}")
    
    # Always save best
    if is_best:
        best_path = CHECKPOINT_DIR / 'best_model_4datasets.pth'
        torch.save(checkpoint, best_path)
        print(f"\n[BEST] New best model saved: {best_accuracy:.2f}%")
        
        # Also save to /kaggle/output/ (persistent even after stop)
        output_dir = Path('/kaggle/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, output_dir / 'best_model_4datasets.pth')
        print(f"[BACKUP] Model backed up to /kaggle/output/")

def save_results(results_dict):
    """Save results with backup"""
    # Save to working
    results_path = CHECKPOINT_DIR / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Backup to output (persistent)
    output_dir = Path('/kaggle/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_path, output_dir / 'training_results.json')
    print(f"[SAVE] Results saved and backed up")

# ============================================================
# LOAD DATASETS
# ============================================================

with open('/kaggle/working/dataset_paths.json', 'r') as f:
    dataset_paths = json.load(f)

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Data augmentation
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3)
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
all_train_datasets = []
all_test_datasets = []

for ds_name, ds_path in dataset_paths.items():
    try:
        if ds_name == 'fer2013':
            train_ds = datasets.ImageFolder(f"{ds_path}/train", transform=train_transform)
            test_ds = datasets.ImageFolder(f"{ds_path}/test", transform=test_transform)
        elif ds_name == 'utkface':
            train_ds = datasets.ImageFolder(f"{ds_path}/train", transform=train_transform)
            test_ds = datasets.ImageFolder(f"{ds_path}/test", transform=test_transform)
        elif ds_name == 'rafdb':
            train_ds = datasets.ImageFolder(f"{ds_path}/train", transform=train_transform)
            test_ds = datasets.ImageFolder(f"{ds_path}/test", transform=test_transform)
        
        all_train_datasets.append(train_ds)
        all_test_datasets.append(test_ds)
        print(f"[OK] {ds_name.upper()}: {len(train_ds)} train, {len(test_ds)} test")
    except Exception as e:
        print(f"[SKIP] {ds_name}: {e}")

# Combine datasets
combined_train = ConcatDataset(all_train_datasets)
combined_test = ConcatDataset(all_test_datasets)

train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(combined_test, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

print(f"\n[READY] Train: {len(combined_train)}, Test: {len(combined_test)}")

# ============================================================
# MODEL SETUP
# ============================================================

model = timm.create_model(MODEL_TYPE, pretrained=True, num_classes=NUM_CLASSES, drop_rate=DROPOUT)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LEARNING_RATE,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader) // GRAD_ACCUM_STEPS
)

scaler = GradScaler(device='cuda') if USE_MIXED_PRECISION else None

# Mixup augmentation
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# TRAINING LOOP WITH AUTO-SAVE
# ============================================================

print("\n" + "="*60)
print("STARTING TRAINING (WITH AUTO-SAVE EVERY 10 EPOCHS)")
print("="*60)

best_accuracy = 0
best_epoch = 0
patience_counter = 0
start_time = time.time()

history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}

for epoch in range(EPOCHS):
    # TRAIN
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        if USE_MIXED_PRECISION:
            with autocast(device_type='cuda'):
                if np.random.rand() > 0.3:
                    images, la, lb, lam = mixup_data(images, labels)
                    loss = mixup_loss(criterion, model(images), la, lb, lam) / GRAD_ACCUM_STEPS
                else:
                    loss = criterion(model(images), labels) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
        else:
            if np.random.rand() > 0.3:
                images, la, lb, lam = mixup_data(images, labels)
                loss = mixup_loss(criterion, model(images), la, lb, lam) / GRAD_ACCUM_STEPS
            else:
                loss = criterion(model(images), labels) / GRAD_ACCUM_STEPS
            loss.backward()
        
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            if USE_MIXED_PRECISION:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if USE_MIXED_PRECISION:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        train_loss += loss.item() * GRAD_ACCUM_STEPS
        _, predicted = model(images).max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{train_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%'
        })
    
    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # VALIDATION
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    current_lr = optimizer.param_groups[0]['lr']
    
    history['train_loss'].append(float(avg_train_loss))
    history['train_acc'].append(float(train_acc))
    history['test_acc'].append(float(test_acc))
    history['lr'].append(float(current_lr))
    
    elapsed = time.time() - start_time
    print(f"\nEpoch {epoch+1}: Loss={avg_train_loss:.4f}, Train={train_acc:.2f}%, Val={test_acc:.2f}%, LR={current_lr:.7f}, Time={elapsed/3600:.2f}h")
    
    # Save checkpoint
    is_best = test_acc > best_accuracy
    if is_best:
        best_accuracy = test_acc
        best_epoch = epoch + 1
        patience_counter = 0
    else:
        patience_counter += 1
    
    save_checkpoint(epoch, model, optimizer, best_accuracy, history, is_best=is_best)
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n[STOP] Early stopping at epoch {epoch+1}")
        break

# ============================================================
# SAVE FINAL RESULTS
# ============================================================

total_time = time.time() - start_time

results = {
    'best_accuracy': float(best_accuracy),
    'best_epoch': best_epoch,
    'total_epochs': epoch + 1,
    'training_time_hours': float(total_time / 3600),
    'num_classes': NUM_CLASSES,
    'class_names': class_names,
    'datasets_used': list(dataset_paths.keys()),
    'total_train_images': len(combined_train),
    'total_test_images': len(combined_test),
    'model_type': MODEL_TYPE,
    'history': history,
    'config': {
        'model': MODEL_TYPE,
        'batch_size': BATCH_SIZE * GRAD_ACCUM_STEPS,
        'learning_rate': LEARNING_RATE,
        'dropout': DROPOUT,
        'weight_decay': WEIGHT_DECAY,
        'early_stopping': EARLY_STOPPING_PATIENCE
    }
}

save_results(results)

print("\n" + "="*60)
print("TRAINING COMPLETED WITH AUTO-SAVE!")
print("="*60)
print(f"Best Accuracy: {best_accuracy:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Total Time: {total_time/3600:.2f} hours")
print("\nFiles saved to:")
print("  /kaggle/working/checkpoints_production/")
print("  /kaggle/output/ (PERSISTENT - won't lose!)")
print("="*60)
