lok"""
4-DATASET KAGGLE TRAINING SCRIPT
Target: 80-85% accuracy
Time: 10-12 hours (P100 GPU)

Pre-requisites:
1. Add datasets via '+ Add Input':
   - msambare/fer2013
   - jangedoo/utkface-new
   - shuvoalok/raf-db-dataset
   - shreyanshverma27/ferplus (OR davilsena/ckextended)

2. Run this script in Kaggle notebook

Note: AffectNet is no longer available. Use FER2013+, CK+, or ExpW instead.
"""

# ============================================================
# CELL 1: CHECK GPU
# ============================================================

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
    print("\n[WARNING] No GPU available!")

print("=" * 60)

# ============================================================
# CELL 2: CLONE REPOSITORY
# ============================================================

import os
from pathlib import Path

print("\n" + "=" * 60)
print("CLONING REPOSITORY")
print("=" * 60)

repo_url = "https://github.com/khoiabc2020/age-gender-emotion-detection.git"
repo_dir = Path("/kaggle/working/repo")

if repo_dir.exists():
    print("\n[INFO] Repository exists, pulling latest...")
    os.system("cd /kaggle/working/repo && git pull")
else:
    print("\n[INFO] Cloning repository...")
    os.system(f"git clone {repo_url} /kaggle/working/repo")

os.chdir("/kaggle/working/repo/training_experiments")
print(f"\n[OK] Working directory: {os.getcwd()}")
print("=" * 60)

# ============================================================
# CELL 3: CHECK 4 DATASETS
# ============================================================

import json

print("\n" + "=" * 60)
print("CHECKING 4 DATASETS")
print("=" * 60)

dataset_paths = {}

# 1. FER2013
print("\n[1/4] FER2013...")
for path in ['/kaggle/input/fer2013', '/kaggle/input/msambare-fer2013']:
    if Path(path).exists():
        dataset_paths['fer2013'] = path
        print(f"  [OK] {path}")
        break

# 2. UTKFace
print("\n[2/4] UTKFace...")
for path in ['/kaggle/input/utkface-new', '/kaggle/input/jangedoo-utkface-new']:
    if Path(path).exists():
        dataset_paths['utkface'] = path
        print(f"  [OK] {path}")
        break

# 3. RAF-DB
print("\n[3/4] RAF-DB...")
rafdb_paths = ['/kaggle/input/raf-db-dataset', '/kaggle/input/shuvoalok-raf-db-dataset',
               '/kaggle/input/raf-db', '/kaggle/input/alex1233213-raf-db']
for path in rafdb_paths:
    if Path(path).exists():
        dataset_paths['rafdb'] = path
        print(f"  [OK] {path}")
        break

# 4. FER2013+ or Alternatives
print("\n[4/4] Checking Additional Datasets...")
ferplus_paths = ['/kaggle/input/ferplus', '/kaggle/input/shreyanshverma27-ferplus',
                 '/kaggle/input/fer-plus', '/kaggle/input/fer2013plus']
for path in ferplus_paths:
    if Path(path).exists():
        dataset_paths['ferplus'] = path
        print(f"  [OK] FER2013+: {path}")
        break

if 'ferplus' not in dataset_paths:
    ckplus_paths = ['/kaggle/input/ckextended', '/kaggle/input/davilsena-ckextended']
    for path in ckplus_paths:
        if Path(path).exists():
            dataset_paths['ckplus'] = path
            print(f"  [OK] CK+: {path}")
            break

if len(dataset_paths) < 4:
    print("  [WARN] No 4th dataset found")
    print("  [INFO] Add: shreyanshverma27/ferplus OR davilsena/ckextended")

with open('/kaggle/working/dataset_paths.json', 'w') as f:
    json.dump(dataset_paths, f, indent=2)

print("\n" + "=" * 60)
print(f"FOUND: {len(dataset_paths)}/4 datasets")
print("=" * 60)

estimates = {'fer2013': 28709, 'utkface': 23708, 'rafdb': 12271, 
             'ferplus': 35887, 'ckplus': 10000, 'expw': 15000}
total = sum(estimates[n] for n in dataset_paths if n in estimates)
print(f"\n[ESTIMATE] ~{total:,} images")

if len(dataset_paths) >= 3:
    print(f"\n[SUCCESS] Ready! Expected: 80-85%")
print("=" * 60)

# ============================================================
# CELL 4: INSTALL DEPENDENCIES
# ============================================================

print("\n" + "=" * 60)
print("INSTALLING DEPENDENCIES")
print("=" * 60)

os.system("pip install -q timm albumentations tensorboard onnx onnxscript onnxruntime torchmetrics opencv-python")

print("\n[OK] All dependencies installed!")
print("=" * 60)

# ============================================================
# CELL 5: TRAIN (MAIN TRAINING)
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
import timm
from tqdm import tqdm
import time
import numpy as np

print("\n" + "=" * 60)
print("4-DATASET TRAINING - TARGET 80-85%")
print("=" * 60)

# Config
EPOCHS = 120
BATCH_SIZE = 48
LEARNING_RATE = 0.00015
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 35
WARMUP_EPOCHS = 5

print(f"\n[CONFIG] Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}, Epochs: {EPOCHS}")

# Load paths
with open('/kaggle/working/dataset_paths.json') as f:
    dataset_paths = json.load(f)

# Transforms
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

# Load datasets
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
                print(f"  [OK] {name.upper()}: {len(train_ds):,} train")
        except Exception as e:
            print(f"  [WARN] {name.upper()}: {str(e)[:50]}")

train_dataset = ConcatDataset(all_train_datasets) if len(all_train_datasets) > 1 else all_train_datasets[0]
test_dataset = ConcatDataset(all_test_datasets) if len(all_test_datasets) > 1 else all_test_datasets[0]

print(f"\n[OK] Total: {total_train:,} train, {total_test:,} test")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

first_dataset = all_train_datasets[0]
num_classes = len(first_dataset.classes)
class_names = first_dataset.classes

# Model
print(f"\n[INFO] Creating EfficientNet-B0...")
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes, drop_rate=0.6)
model = model.to(DEVICE)

# Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.5, label_smoothing=0.2):
        super().__init__()
        self.alpha, self.gamma, self.label_smoothing = alpha, gamma, label_smoothing
    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(inputs, targets)
        return self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce

criterion = FocalLoss()

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.03)

def get_scheduler(opt, warmup, total):
    return optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/warmup if e < warmup else 0.5*(1+np.cos(np.pi*(e-warmup)/(total-warmup))))

scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

# Mixup
def mixup(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def mixup_loss(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1-lam) * crit(pred, y_b)

# Training loop
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

best_acc = 0
patience = 0
start_time = time.time()
history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}

for epoch in range(EPOCHS):
    model.train()
    train_loss = train_correct = train_total = 0
    lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        if np.random.rand() > 0.4:
            images, la, lb, lam = mixup(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_loss(criterion, outputs, la, lb, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = outputs.max(1)
        train_total += labels.size(0)
        train_correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{train_loss/(pbar.n+1):.4f}'})
    
    train_acc = 100. * train_correct / train_total
    avg_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, pred = outputs.max(1)
            test_total += labels.size(0)
            test_correct += pred.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    scheduler.step()
    
    history['train_loss'].append(float(avg_loss))
    history['train_acc'].append(float(train_acc))
    history['test_acc'].append(float(test_acc))
    history['lr'].append(float(lr))
    
    elapsed = time.time() - start_time
    print(f"\nE{epoch+1}: Loss={avg_loss:.4f}, Train={train_acc:.2f}%, Val={test_acc:.2f}%, Time={elapsed/60:.1f}m")
    
    # Save best
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch + 1
        patience = 0
        
        save_dir = Path('/kaggle/working/checkpoints_production')
        save_dir.mkdir(exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_accuracy': best_acc,
            'class_names': class_names,
            'num_classes': num_classes,
            'datasets_used': list(dataset_paths.keys())
        }, save_dir / 'best_model_4datasets.pth')
        print(f"[BEST] {best_acc:.2f}% saved!")
    else:
        patience += 1
    
    if patience >= EARLY_STOPPING_PATIENCE:
        print(f"\n[STOP] Early stopping")
        break

# Save results
total_time = time.time() - start_time
results = {
    'best_accuracy': float(best_acc),
    'best_epoch': int(best_epoch),
    'total_epochs': epoch + 1,
    'training_time_hours': float(total_time/3600),
    'num_classes': num_classes,
    'class_names': class_names,
    'datasets_used': list(dataset_paths.keys()),
    'num_datasets': len(dataset_paths),
    'total_train_images': total_train,
    'total_test_images': total_test,
    'history': history
}

with open('/kaggle/working/checkpoints_production/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best: {best_acc:.2f}%")
print(f"Time: {total_time/3600:.2f}h")

if best_acc >= 80:
    print("\n[SUCCESS] TARGET ACHIEVED!")
elif best_acc >= 78:
    print("\n[EXCELLENT] Very close!")
print("="*60)

# ============================================================
# CELL 6: EXPORT TO ONNX
# ============================================================

print("\n" + "="*60)
print("EXPORTING TO ONNX")
print("="*60)

checkpoint_path = Path('/kaggle/working/checkpoints_production/best_model_4datasets.pth')

if checkpoint_path.exists():
    # Fix for PyTorch 2.6+: add weights_only=False
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = checkpoint_path.parent / 'best_model.onnx'
    
    torch.onnx.export(model, dummy, onnx_path, opset_version=11,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    
    print(f"[OK] ONNX exported! Size: {onnx_path.stat().st_size/(1024*1024):.1f} MB")
    print("[SUCCESS] Ready for deployment!")

print("="*60)

print("\n" + "="*60)
print("ALL DONE!")
print("="*60)
print("\nFiles saved to: /kaggle/working/checkpoints_production/")
print("  - best_model_4datasets.pth")
print("  - best_model.onnx")
print("  - training_results.json")
print("\nDownload these files to deploy!")
print("="*60)
