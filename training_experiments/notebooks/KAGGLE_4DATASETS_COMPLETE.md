# Complete 9-Cell Guide for 4-Dataset Training (80-85%)

## Target: 80-85% Accuracy
## Datasets: FER2013 + UTKFace + RAF-DB + AffectNet
## Time: 10-12 hours (P100 GPU)

---

## CELL 1: CHECK GPU

```python
import torch
import torch.cuda as cuda

print("=" * 60)
print("CHECKING GPU")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {cuda.is_available()}")

if cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {cuda.get_device_name(0)}")
    print(f"GPU memory: {cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n[OK] GPU is ready!")
else:
    print("\n[WARNING] No GPU available! Training will be very slow.")

print("=" * 60)
```

---

## CELL 2: CLONE REPOSITORY

```python
import os
from pathlib import Path

print("=" * 60)
print("CLONING REPOSITORY")
print("=" * 60)

repo_url = "https://github.com/khoiabc2020/age-gender-emotion-detection.git"
repo_dir = Path("/kaggle/working/repo")

if repo_dir.exists():
    print("\n[INFO] Repository exists, pulling latest changes...")
    !cd /kaggle/working/repo && git pull
    print("[OK] Updated to latest version")
else:
    print("\n[INFO] Cloning repository...")
    !git clone {repo_url} /kaggle/working/repo
    print("[OK] Repository cloned")

%cd /kaggle/working/repo/training_experiments

print(f"\n[OK] Working directory: {os.getcwd()}")
print("=" * 60)
```

---

## CELL 3: CHECK 4 DATASETS

⚠️ **IMPORTANT:** Make sure you've added all 4 datasets via "+ Add Input" first!

```python
import json
from pathlib import Path

print("=" * 60)
print("CHECKING 4 DATASETS")
print("=" * 60)

dataset_paths = {}

# 1. FER2013 - Main emotion dataset (28K images)
print("\n[1/4] Checking FER2013...")
fer_paths = [
    '/kaggle/input/fer2013',
    '/kaggle/input/msambare-fer2013'
]
for path in fer_paths:
    if Path(path).exists():
        dataset_paths['fer2013'] = path
        print(f"  [OK] FER2013: {path}")
        break
if 'fer2013' not in dataset_paths:
    print("  [ERROR] FER2013 not found! This is required.")

# 2. UTKFace - Age/Gender dataset (23K images)
print("\n[2/4] Checking UTKFace...")
utk_paths = [
    '/kaggle/input/utkface-new',
    '/kaggle/input/jangedoo-utkface-new'
]
for path in utk_paths:
    if Path(path).exists():
        dataset_paths['utkface'] = path
        print(f"  [OK] UTKFace: {path}")
        break
if 'utkface' not in dataset_paths:
    print("  [WARN] UTKFace not found")
    print("  [INFO] Add via: + Add Input -> Search 'jangedoo/utkface-new'")

# 3. RAF-DB - High-quality emotion dataset (12K images)
print("\n[3/4] Checking RAF-DB...")
rafdb_paths = [
    '/kaggle/input/raf-db-dataset',
    '/kaggle/input/shuvoalok-raf-db-dataset',
    '/kaggle/input/raf-db',
    '/kaggle/input/alex1233213-raf-db'
]
for path in rafdb_paths:
    if Path(path).exists():
        dataset_paths['rafdb'] = path
        print(f"  [OK] RAF-DB: {path}")
        break
if 'rafdb' not in dataset_paths:
    print("  [WARN] RAF-DB not found")
    print("  [INFO] Add via: + Add Input -> Search 'shuvoalok/raf-db-dataset'")

# 4. AffectNet - Large-scale emotion dataset (30K images)
print("\n[4/4] Checking AffectNet...")
affectnet_paths = [
    '/kaggle/input/affectnet-training-data',
    '/kaggle/input/noamsegal-affectnet-training-data',
    '/kaggle/input/affectnet-cnn-validation',
    '/kaggle/input/tom99763-affectnet-cnn-validation'
]
for path in affectnet_paths:
    if Path(path).exists():
        dataset_paths['affectnet'] = path
        print(f"  [OK] AffectNet: {path}")
        break
if 'affectnet' not in dataset_paths:
    print("  [WARN] AffectNet not found")
    print("  [INFO] Add via: + Add Input -> Search 'noamsegal/affectnet-training-data'")

# Save paths
paths_file = '/kaggle/working/dataset_paths.json'
with open(paths_file, 'w') as f:
    json.dump(dataset_paths, f, indent=2)

print("\n" + "=" * 60)
print(f"DATASETS READY: {len(dataset_paths)}/4")
print("=" * 60)

for name, path in dataset_paths.items():
    print(f"  {name.upper()}: {path}")

print(f"\n[INFO] Paths saved to: {paths_file}")

# Estimate total images
estimates = {
    'fer2013': 28709,
    'utkface': 23708,
    'rafdb': 12271,
    'affectnet': 30000
}

total_estimate = sum(estimates[name] for name in dataset_paths.keys() if name in estimates)
print(f"\n[ESTIMATE] Total images: ~{total_estimate:,}")

if len(dataset_paths) >= 3:
    print("\n[SUCCESS] Ready for high-accuracy training!")
    print(f"Expected accuracy: 80-85%")
elif len(dataset_paths) >= 2:
    print("\n[OK] Ready for training with 2 datasets")
    print(f"Expected accuracy: 77-82%")
else:
    print("\n[WARN] Only 1 dataset found")
    print(f"Expected accuracy: 75-80%")
    print("\nTo reach 80-85%, please add more datasets via '+ Add Input'")

print("=" * 60)
```

---

## CELL 4: INSTALL DEPENDENCIES

```python
print("=" * 60)
print("INSTALLING DEPENDENCIES")
print("=" * 60)

print("\n[INFO] Installing packages...")
print("[INFO] Time: ~2-3 minutes\n")

%pip install -q timm albumentations tensorboard onnx onnxscript onnxruntime torchmetrics opencv-python

print("\n[OK] All dependencies installed!")
print("=" * 60)
```

---

## CELL 5: TRAIN WITH 4 DATASETS (Main Training Cell)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import timm
import json
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

print("=" * 60)
print("4-DATASET TRAINING - TARGET 80-85%")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 120
BATCH_SIZE = 48
LEARNING_RATE = 0.00015
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 35
WARMUP_EPOCHS = 5

print(f"\n[CONFIG] Training Configuration:")
print(f"  Model: EfficientNet-B0")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Early Stop Patience: {EARLY_STOPPING_PATIENCE}")
print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ============================================================
# LOAD DATASET PATHS
# ============================================================

with open('/kaggle/working/dataset_paths.json') as f:
    dataset_paths = json.load(f)

print(f"\n[INFO] Found {len(dataset_paths)} datasets:")
for name in dataset_paths.keys():
    print(f"  - {name.upper()}")

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.25)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# LOAD ALL DATASETS
# ============================================================

print("\n[INFO] Loading datasets...")

all_train_datasets = []
all_test_datasets = []
total_train = 0
total_test = 0

for name, path in dataset_paths.items():
    dataset_path = Path(path)
    
    # Try different directory structures
    train_dir = None
    test_dir = None
    
    # Structure 1: path/train, path/test
    if (dataset_path / 'train').exists():
        train_dir = dataset_path / 'train'
        test_dir = dataset_path / 'test' if (dataset_path / 'test').exists() else train_dir
    else:
        # Structure 2: Search for train/test subdirectories
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
            
            if len(train_ds.classes) > 0 and len(train_ds) > 0:
                all_train_datasets.append(train_ds)
                all_test_datasets.append(test_ds)
                total_train += len(train_ds)
                total_test += len(test_ds)
                print(f"  [OK] {name.upper()}: {len(train_ds):,} train, {len(test_ds):,} test, {len(train_ds.classes)} classes")
        except Exception as e:
            print(f"  [WARN] {name.upper()}: Failed - {str(e)[:50]}")
    else:
        print(f"  [WARN] {name.upper()}: train directory not found")

# Combine datasets
if len(all_train_datasets) == 0:
    print("\n[ERROR] No datasets loaded!")
    raise RuntimeError("No valid datasets found. Check '+ Add Input'")

train_dataset = ConcatDataset(all_train_datasets) if len(all_train_datasets) > 1 else all_train_datasets[0]
test_dataset = ConcatDataset(all_test_datasets) if len(all_test_datasets) > 1 else all_test_datasets[0]

print(f"\n[SUCCESS] Combined datasets:")
print(f"  Total train: {total_train:,} images")
print(f"  Total test: {total_test:,} images")
print(f"  Datasets used: {len(all_train_datasets)}")

# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Get num_classes from first dataset
first_dataset = all_train_datasets[0]
num_classes = len(first_dataset.classes) if hasattr(first_dataset, 'classes') else 7
class_names = first_dataset.classes if hasattr(first_dataset, 'classes') else [f'class_{i}' for i in range(num_classes)]

print(f"[OK] Classes: {num_classes}")
print(f"[OK] Batches: {len(train_loader)} train, {len(test_loader)} test")

# ============================================================
# MODEL
# ============================================================

print("\n[INFO] Creating EfficientNet-B0 with dropout=0.6...")
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes, drop_rate=0.6)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] Model ready ({total_params:,} parameters)")

# ============================================================
# LOSS FUNCTION
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.5, label_smoothing=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss

criterion = FocalLoss(alpha=1, gamma=2.5, label_smoothing=0.2)
print("[OK] Focal Loss (gamma=2.5, smoothing=0.2)")

# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.03, betas=(0.9, 0.999))

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
print("[OK] AdamW + Cosine Annealing with Warmup")

# ============================================================
# MIXUP AUGMENTATION
# ============================================================

def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print("[OK] Mixup (alpha=0.3)")

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print(f"Expected: 80-85% with {len(all_train_datasets)} datasets")
print("=" * 60)

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
    
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Mixup 60% of batches
        if np.random.rand() > 0.4:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.3)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{train_loss/(batch_idx+1):.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
    
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
    
    # Update scheduler
    scheduler.step()
    
    # Save history
    history['train_loss'].append(float(avg_train_loss))
    history['train_acc'].append(float(train_acc))
    history['test_acc'].append(float(test_acc))
    history['lr'].append(float(current_lr))
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\nEpoch {epoch+1}: Loss={avg_train_loss:.4f}, Train={train_acc:.2f}%, Val={test_acc:.2f}%, LR={current_lr:.7f}, Time={elapsed/60:.1f}m")
    
    # Save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_epoch = epoch + 1
        patience_counter = 0
        
        save_dir = Path('/kaggle/working/checkpoints_production')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'class_names': class_names,
            'num_classes': num_classes,
            'datasets_used': list(dataset_paths.keys()),
            'config': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'model': 'efficientnet_b0',
                'dropout': 0.6,
                'num_datasets': len(dataset_paths)
            }
        }
        
        torch.save(checkpoint, save_dir / 'best_model_4datasets.pth')
        print(f"[NEW BEST] {best_accuracy:.2f}% saved!")
    else:
        patience_counter += 1
        print(f"No improvement: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n[STOP] Early stopping at epoch {epoch+1}")
        break

# ============================================================
# SAVE RESULTS
# ============================================================

total_time = time.time() - start_time

results = {
    'best_accuracy': float(best_accuracy),
    'best_epoch': int(best_epoch),
    'total_epochs': epoch + 1,
    'training_time_hours': float(total_time/3600),
    'num_classes': num_classes,
    'class_names': class_names,
    'datasets_used': list(dataset_paths.keys()),
    'num_datasets': len(dataset_paths),
    'total_train_images': total_train,
    'total_test_images': total_test,
    'history': history,
    'config': {
        'model': 'efficientnet_b0',
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'dropout': 0.6,
        'label_smoothing': 0.2,
        'focal_gamma': 2.5,
        'num_datasets': len(dataset_paths)
    }
}

save_dir = Path('/kaggle/working/checkpoints_production')
save_dir.mkdir(parents=True, exist_ok=True)

results_path = save_dir / 'training_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nBest Accuracy: {best_accuracy:.2f}%")
print(f"Best Epoch: {best_epoch}/{epoch+1}")
print(f"Datasets Used: {len(dataset_paths)}")
print(f"Total Images: {total_train:,} train + {total_test:,} test")
print(f"Training Time: {total_time/3600:.2f} hours")

if best_accuracy >= 80:
    print("\n[SUCCESS] TARGET ACHIEVED! (80-85%)")
    print("Model is production-ready!")
elif best_accuracy >= 78:
    print("\n[EXCELLENT] Very close! (78-80%)")
    print("Model is near-production ready!")
elif best_accuracy >= 75:
    print("\n[GOOD] Good performance! (75-78%)")
else:
    print(f"\n[OK] Completed with {best_accuracy:.2f}%")

print(f"\n[SAVED] Model: {save_dir / 'best_model_4datasets.pth'}")
print(f"[SAVED] Results: {results_path}")
print("=" * 60)
```

---

## CELL 6: CHECK RESULTS

```python
import json
from pathlib import Path

print("=" * 60)
print("TRAINING RESULTS")
print("=" * 60)

# Check multiple possible file locations
possible_paths = [
    Path('/kaggle/working/checkpoints_production/training_results.json'),
    Path('/kaggle/working/checkpoints_production/training_results_improved.json'),
    Path('/kaggle/working/training_results.json')
]

results_file = None
for path in possible_paths:
    if path.exists():
        results_file = path
        break

if results_file:
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"\n[SUCCESS] Training Completed!")
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    print(f"Best Accuracy: {results.get('best_accuracy', 0):.2f}%")
    print(f"Best Epoch: {results.get('best_epoch', 'N/A')}")
    print(f"Total Epochs: {results.get('total_epochs', 'N/A')}")
    
    if 'training_time_hours' in results:
        print(f"Training Time: {results['training_time_hours']:.2f} hours")
    
    if 'datasets_used' in results:
        print(f"\nDatasets Used: {len(results['datasets_used'])}")
        for ds in results['datasets_used']:
            print(f"  - {ds.upper()}")
    
    if 'total_train_images' in results:
        print(f"\nTotal Images:")
        print(f"  Train: {results['total_train_images']:,}")
        print(f"  Test: {results['total_test_images']:,}")
    
    # Evaluation
    best_acc = results.get('best_accuracy', 0)
    print(f"\n{'='*60}")
    print("EVALUATION")
    print('='*60)
    
    if best_acc >= 80:
        print("[SUCCESS] TARGET ACHIEVED! (80-85%)")
        print("Model is PRODUCTION-READY!")
    elif best_acc >= 78:
        print("[EXCELLENT] Very close! (78-80%)")
        print("Model is near-production ready!")
    elif best_acc >= 75:
        print("[GOOD] Good performance! (75-78%)")
        print("Model can be used in production with monitoring")
    elif best_acc >= 70:
        print("[OK] Decent performance (70-75%)")
        print("Consider adding more data or training longer")
    else:
        print(f"[INFO] Completed with {best_acc:.2f}%")
    
    # Show config
    if 'config' in results:
        print(f"\n{'='*60}")
        print("CONFIGURATION")
        print('='*60)
        for key, value in results['config'].items():
            print(f"  {key}: {value}")
    
    print(f"\n[INFO] Results file: {results_file}")
    
else:
    print("\n[ERROR] Results file not found!")
    print("\nSearched locations:")
    for path in possible_paths:
        print(f"  - {path} {'[EXISTS]' if path.exists() else '[NOT FOUND]'}")
    print("\n[INFO] Training may still be in progress or failed")
    print("[INFO] Check Cell 5 output for errors")

print("=" * 60)
```

---

## CELL 7: EXPORT TO ONNX

```python
import torch
from pathlib import Path
import sys

print("=" * 60)
print("EXPORTING TO ONNX")
print("=" * 60)

# Find checkpoint
checkpoint_names = [
    'best_model_4datasets.pth',
    'best_model_optimized.pth',
    'best_model_improved.pth',
    'best_model.pth'
]

checkpoint_path = None
for name in checkpoint_names:
    path = Path(f'/kaggle/working/checkpoints_production/{name}')
    if path.exists():
        checkpoint_path = path
        break

if not checkpoint_path:
    print("[ERROR] Model checkpoint not found!")
    print("\nSearched for:")
    for name in checkpoint_names:
        print(f"  - {name}")
    print("\n[INFO] Please complete training (Cell 5) first")
else:
    print(f"\n[INFO] Loading: {checkpoint_path.name}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Import model
        import timm
        num_classes = checkpoint['num_classes']
        
        print(f"[INFO] Creating model (classes={num_classes})...")
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to('cpu')
        
        print(f"[OK] Model loaded (acc: {checkpoint['best_accuracy']:.2f}%)")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        
        print("\n[INFO] Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            print(f"[OK] Output shape: {output.shape}")
        
        # Export to ONNX
        onnx_path = checkpoint_path.parent / 'best_model.onnx'
        
        print(f"\n[INFO] Exporting to: {onnx_path.name}")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        size_mb = onnx_path.stat().st_size / (1024*1024)
        print(f"[OK] ONNX exported! Size: {size_mb:.1f} MB")
        
        # Verify ONNX
        print("\n[INFO] Verifying ONNX model...")
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid!")
        
        # Test with ONNX Runtime
        print("\n[INFO] Testing with ONNX Runtime...")
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(str(onnx_path))
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        ort_outputs = session.run(None, {'input': test_input})
        
        print(f"[OK] ONNX Runtime test passed!")
        print(f"[INFO] Output shape: {ort_outputs[0].shape}")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] MODEL READY FOR DEPLOYMENT!")
        print("=" * 60)
        print(f"\nFiles:")
        print(f"  PyTorch: {checkpoint_path.name} ({checkpoint_path.stat().st_size/(1024*1024):.1f} MB)")
        print(f"  ONNX: {onnx_path.name} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()

print("=" * 60)
```

---

## CELL 8: DOWNLOAD FILES

```python
from IPython.display import FileLink, display
from pathlib import Path

print("=" * 60)
print("DOWNLOAD TRAINED MODELS")
print("=" * 60)

checkpoint_dir = Path('/kaggle/working/checkpoints_production')

if not checkpoint_dir.exists():
    print("[ERROR] Checkpoint directory not found!")
    print("[INFO] Please complete training (Cell 5) first")
else:
    files = list(checkpoint_dir.glob('*'))
    
    if not files:
        print("[WARN] No files found in checkpoint directory")
    else:
        print(f"\n[INFO] Found {len(files)} files:\n")
        
        total_size = 0
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size / (1024*1024)
                total_size += size
                print(f"{file_path.name} ({size:.1f} MB):")
                display(FileLink(str(file_path)))
                print()
        
        print(f"Total size: {total_size:.1f} MB")
        print("\n[INFO] Click links above to download!")
        print("\n[NEXT STEPS]")
        print("1. Download all files")
        print("2. Deploy to your project:")
        print("   - best_model_*.pth -> training_experiments/checkpoints/production/")
        print("   - best_model.onnx -> ai_edge_app/models/")
        print("   - training_results.json -> training_experiments/results/")

print("=" * 60)
```

---

## CELL 9: TRAINING SUMMARY

```python
import json
from pathlib import Path

print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

results_file = Path('/kaggle/working/checkpoints_production/training_results.json')

if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print('='*60)
    
    print(f"\nAccuracy: {results['best_accuracy']:.2f}%")
    print(f"Training Time: {results['training_time_hours']:.2f} hours")
    print(f"Best Epoch: {results['best_epoch']}/{results['total_epochs']}")
    
    if 'datasets_used' in results:
        print(f"\nDatasets ({len(results['datasets_used'])}):")
        for ds in results['datasets_used']:
            print(f"  - {ds.upper()}")
    
    if 'total_train_images' in results:
        print(f"\nImages:")
        print(f"  Train: {results['total_train_images']:,}")
        print(f"  Test: {results['total_test_images']:,}")
    
    print(f"\n{'='*60}")
    print("FILES READY")
    print('='*60)
    
    checkpoint_dir = Path('/kaggle/working/checkpoints_production')
    for file in checkpoint_dir.glob('*'):
        if file.is_file():
            size = file.stat().st_size / (1024*1024)
            print(f"  {file.name} ({size:.1f} MB)")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print('='*60)
    print("1. Download files (Cell 8)")
    print("2. Deploy to project:")
    print("   cd 'D:\\AI vietnam\\Code\\nhan dien do tuoi'")
    print("   # Copy files to correct locations")
    print("3. Test model locally:")
    print("   cd ai_edge_app")
    print("   python main.py")
    print("4. Deploy to production")
    
    print(f"\n{'='*60}")
    
    # Final verdict
    best_acc = results['best_accuracy']
    if best_acc >= 80:
        print("STATUS: PRODUCTION READY! ✓")
    elif best_acc >= 78:
        print("STATUS: NEAR PRODUCTION READY")
    elif best_acc >= 75:
        print("STATUS: GOOD FOR TESTING")
    else:
        print(f"STATUS: COMPLETED ({best_acc:.2f}%)")
    
    print('='*60)
    
else:
    print("\n[ERROR] No results found")
    print("[INFO] Please run Cell 5 (Training) first")

print("=" * 60)
```

---

## 🎯 USAGE

1. **Add datasets in Kaggle** (+ Add Input):
   - FER2013 (required)
   - UTKFace (recommended)
   - RAF-DB (recommended)
   - AffectNet (recommended)

2. **Copy cells 1-9** into your Kaggle notebook

3. **Run cells in order** (1 → 9)

4. **Wait for training** (~10-12 hours)

5. **Download models** (Cell 8)

---

## 📊 EXPECTED RESULTS

| Datasets | Expected Accuracy | Training Time |
|----------|-------------------|---------------|
| 1 | 75-80% | 6-8h |
| 2 | 77-82% | 8-10h |
| 3-4 | **80-85%** ✓ | 10-12h |

---

**All 9 cells ready to copy!** 🚀
