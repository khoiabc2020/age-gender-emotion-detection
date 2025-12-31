# Train on Kaggle Notebooks - Free Alternative to Colab

## Why Kaggle?

- **30 hours GPU/week** (vs Colab 12 hours)
- **GPU: T4 or P100** (better than Colab T4)
- **No disconnection issues**
- **Datasets already available**
- **100% FREE**

---

## Setup Guide

### Step 1: Create Kaggle Account
1. Go to: https://www.kaggle.com/
2. Sign up (free)
3. Go to Settings → Create New API Token → Download `kaggle.json`

### Step 2: Create New Notebook
1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Select **"Notebook"** (not Script)
4. Settings:
   - Accelerator: **GPU T4 x2** or **GPU P100**
   - Internet: **ON**
   - Environment: Python
   - Persistence: Session-only

### Step 3: Add Datasets
1. Click **"+ Add Input"** on right sidebar
2. Search and add:
   - `msambare/fer2013` - Emotion dataset
   - `jangedoo/utkface-new` - Age/Gender dataset
   - `shuvoalok/raf-db-dataset` - RAF-DB (optional)
   - `noamsegal/affectnet-training-data` - AffectNet (optional)

### Step 4: Copy Code from Production Notebook

Use the cells below (copy-paste into Kaggle):

---

## Cell 1: Check GPU

```python
import torch
import torch.cuda as cuda

print("PyTorch version:", torch.__version__)
print("CUDA available:", cuda.is_available())

if cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", cuda.get_device_name(0))
    print(f"GPU memory: {cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: No GPU available!")
```

---

## Cell 2: Clone Repository

```python
import os
from pathlib import Path

repo_url = "https://github.com/khoiabc2020/age-gender-emotion-detection.git"
repo_dir = Path("/kaggle/working/repo")

if repo_dir.exists():
    print("[INFO] Repository exists, pulling latest changes...")
    !cd /kaggle/working/repo && git pull
else:
    print("[INFO] Cloning repository...")
    !git clone {repo_url} /kaggle/working/repo

# Change to project directory
%cd /kaggle/working/repo/training_experiments
print("\n[OK] Repository ready!")
print(f"Working directory: {os.getcwd()}")
```

---

## Cell 3: Locate Datasets

```python
import json
from pathlib import Path

# Kaggle datasets are in /kaggle/input/
dataset_paths = {
    'fer2013': '/kaggle/input/fer2013',
    'utkface': '/kaggle/input/utkface-new'
}

# Check if datasets exist
print("Checking datasets:")
for name, path in dataset_paths.items():
    if Path(path).exists():
        print(f"  [OK] {name.upper()}: {path}")
    else:
        print(f"  [WARN] {name.upper()}: Not found!")
        print(f"         Add it via: + Add Input → Search '{name}'")

# Add optional datasets if available
optional_datasets = {
    'rafdb': '/kaggle/input/raf-db-dataset',
    'affectnet': '/kaggle/input/affectnet-training-data'
}

for name, path in optional_datasets.items():
    if Path(path).exists():
        dataset_paths[name] = path
        print(f"  [OK] {name.upper()} (optional): {path}")

# Save paths to JSON
paths_file = '/kaggle/working/dataset_paths.json'
with open(paths_file, 'w') as f:
    json.dump(dataset_paths, f, indent=2)

print(f"\n[INFO] Total datasets: {len(dataset_paths)}")
print(f"[INFO] Paths saved to: {paths_file}")
```

---

## Cell 4: Install Dependencies

```python
print("Installing production dependencies...")
print("This may take 2-3 minutes\n")

!pip install -q timm albumentations imgaug tensorboard onnx onnxscript onnxruntime torchmetrics opencv-python

print("\n[OK] All dependencies installed!")
```

---

## Cell 5: Run Production Training

```python
import os
from pathlib import Path
import torch

print("=" * 60)
print("PRODUCTION TRAINING - KAGGLE")
print("Target: 78-85% Accuracy")
print("=" * 60)

# Configuration
print("\n[CONFIG] Training Configuration:")
print("  Backbone: EfficientNet-B2")
print("  Epochs: 100 (with early stopping)")
print("  Batch Size: 64")
print("  Learning Rate: 0.0001")
print("  Optimizer: AdamW")
print("  Loss: Focal Loss + Label Smoothing")

# Estimate time
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if 'T4' in gpu_name:
        print("\n[INFO] Using T4 GPU - Estimated time: 10-12 hours")
    elif 'P100' in gpu_name:
        print("\n[INFO] Using P100 GPU - Estimated time: 6-8 hours")

# Verify dataset paths
if not Path('/kaggle/working/dataset_paths.json').exists():
    print("[ERROR] Dataset paths not found! Run Cell 3 first.")
    raise FileNotFoundError("Dataset paths required")

print("\n[START] Starting production training...")
print("=" * 60)
print("\n")

# Navigate to training directory
%cd /kaggle/working/repo/training_experiments

# Run production training
!python train_production.py \
    --data_paths /kaggle/working/dataset_paths.json \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --patience 15 \
    --save_dir /kaggle/working/checkpoints_production

print("\n" + "=" * 60)
print("[OK] TRAINING COMPLETE!")
print("=" * 60)
```

---

## Cell 6: Evaluate Results

```python
import json
from pathlib import Path

print("=" * 60)
print("TRAINING RESULTS")
print("=" * 60)

results_file = Path('/kaggle/working/checkpoints_production/training_results.json')

if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"\n[SUCCESS] Training Completed!")
    print(f"\nBest Validation Accuracy: {results.get('best_accuracy', 0):.2f}%")
    print(f"Best Epoch: {results.get('best_epoch', 'N/A')}")
    print(f"Total Epochs: {results.get('total_epochs', 'N/A')}")
    
    best_acc = results.get('best_accuracy', 0)
    if best_acc >= 78:
        print("\n[OK] TARGET ACHIEVED! (78-85%)")
    elif best_acc >= 75:
        print("\n[OK] Good accuracy, close to target")
    else:
        print("\n[WARN] Below target")
    
    print(f"\n[INFO] Model saved to:")
    print(f"  - /kaggle/working/checkpoints_production/best_model.pth")
else:
    print("\n[WARN] Results file not found")

print("=" * 60)
```

---

## Cell 7: Download Results

```python
from IPython.display import FileLink
from pathlib import Path

print("=" * 60)
print("DOWNLOAD TRAINED MODELS")
print("=" * 60)

files_to_download = [
    '/kaggle/working/checkpoints_production/best_model.pth',
    '/kaggle/working/checkpoints_production/training_results.json'
]

print("\nFiles available for download:")
for file_path in files_to_download:
    if Path(file_path).exists():
        size = Path(file_path).stat().st_size / (1024*1024)
        print(f"\n{Path(file_path).name} ({size:.1f} MB):")
        display(FileLink(file_path))
    else:
        print(f"\n[WARN] {Path(file_path).name} - Not found")

print("\n[INFO] Click links above to download")
print("=" * 60)
```

---

## Key Differences: Kaggle vs Colab

| Feature | Kaggle | Colab |
|---------|--------|-------|
| GPU Time/Week | **30 hours** | 12 hours |
| GPU Type | T4 or **P100** | T4 |
| Disconnection | Rare | Frequent |
| Dataset Location | `/kaggle/input/` | Need download |
| Output Location | `/kaggle/working/` | `/content/` |
| Internet | Need enable | Default ON |

---

## Tips for Kaggle

1. **Enable Internet**: Settings → Internet → ON (required for git clone)
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2 or P100
3. **Add datasets first**: Use "+ Add Input" before running code
4. **Save output**: Use "Save Version" to keep trained models
5. **Monitor usage**: Check GPU quota at https://www.kaggle.com/settings

---

## Troubleshooting

### "Internet is off"
- Click Settings (gear icon) → Turn ON Internet

### "Dataset not found"
- Click "+ Add Input" → Search dataset name → Add

### "GPU quota exceeded"
- Wait until next week (resets weekly)
- Use CPU (slow but works)
- Try other platforms (see below)

---

## Next Steps After Training

1. Run all cells in order (1-7)
2. Wait for training (~10 hours)
3. Check results (Cell 6)
4. Download models (Cell 7)
5. Deploy to your project:
   ```bash
   # Copy to project
   cp best_model.pth training_experiments/checkpoints/production/
   cp training_results.json training_experiments/results/
   ```

---

**Happy Training on Kaggle! 🚀**
