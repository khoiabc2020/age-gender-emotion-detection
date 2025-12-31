# 🚀 SIMPLE COLAB TRAINING - COPY & PASTE

**Copy từng cell dưới đây vào Colab và chạy tuần tự**

---

## 📋 SETUP (Cells 1-3)

### **Cell 1: Check GPU**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### **Cell 2: Mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### **Cell 3: Clone Repository**
```python
import os
if not os.path.exists('/content/repo'):
    !git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git /content/repo
    print("✅ Cloned!")
else:
    print("✅ Already exists!")
%cd /content/repo/training_experiments
!pwd
```

---

## 📥 DOWNLOAD DATA (Cells 4-5)

### **Cell 4: Setup Kaggle**
```python
from google.colab import files
import os, shutil

print("📤 Upload kaggle.json")
print("Get it: https://www.kaggle.com/settings/account → API → Create Token\n")

uploaded = files.upload()

if 'kaggle.json' in uploaded:
    os.makedirs('/root/.kaggle', exist_ok=True)
    shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
    os.chmod('/root/.kaggle/kaggle.json', 600)
    print("\n✅ Kaggle configured!")
```

### **Cell 5: Download Datasets**
```python
!pip install -q kagglehub

import kagglehub

print("📥 Downloading FER2013 (5-10 min)...")
fer2013 = kagglehub.dataset_download("msambare/fer2013")
print(f"✅ FER2013: {fer2013}")

print("\n📥 Downloading UTKFace (5-10 min)...")  
utkface = kagglehub.dataset_download("jangedoo/utkface-new")
print(f"✅ UTKFace: {utkface}")

# Save paths
with open('/content/dataset_paths.txt', 'w') as f:
    f.write(f"{fer2013}\n{utkface}")

print("\n✅ All datasets ready!")
```

---

## 🔧 INSTALL DEPENDENCIES (Cell 6)

### **Cell 6: Install Packages**
```python
%cd /content/repo/training_experiments

!pip install -q albumentations tensorboard onnx onnxruntime

print("✅ Dependencies installed!")
```

---

## 🚀 TRAINING (Cell 7)

### **Cell 7: Start Training**
```python
%cd /content/repo/training_experiments

# Read dataset paths
with open('/content/dataset_paths.txt') as f:
    paths = f.read().strip().split('\n')
    fer2013_path = paths[0]
    utkface_path = paths[1]

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"FER2013: {fer2013_path}")
print(f"UTKFace: {utkface_path}")
print(f"Epochs: 50")
print(f"Batch: 64")
print(f"GPU: Tesla T4")
print("=" * 60)

# Run training
!python train_week2_lightweight.py \
    --data_dir {fer2013_path} \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --save_dir /content/checkpoints

print("\n✅ TRAINING COMPLETED!")
```

**Thời gian**: ~45-60 phút

---

## 💾 SAVE RESULTS (Cell 8)

### **Cell 8: Copy to Drive**
```python
import shutil
from pathlib import Path

src = Path("/content/checkpoints")
dst = Path("/content/drive/MyDrive/SmartRetailAI_Models")
dst.mkdir(parents=True, exist_ok=True)

if src.exists():
    print("📦 Copying to Drive...")
    for file in src.glob("*"):
        if file.is_file():
            shutil.copy(file, dst / file.name)
            size = file.stat().st_size / 1024 / 1024
            print(f"   ✅ {file.name} ({size:.1f} MB)")
    
    print(f"\n✅ Saved to: {dst}")
    print("\n📁 Files in Drive:")
    for f in dst.glob("*"):
        print(f"   - {f.name}")
else:
    print("❌ No checkpoints found!")
```

---

## 🎉 DONE! (Cell 9)

### **Cell 9: Summary**
```python
import json
from pathlib import Path

print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

# Load results if exists
results_file = Path("/content/checkpoints/training_results.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    print(f"\n📊 Results:")
    print(f"   Gender Accuracy: {results.get('gender_acc', 'N/A')}%")
    print(f"   Emotion Accuracy: {results.get('emotion_acc', 'N/A')}%")
    print(f"   Age MAE: {results.get('age_mae', 'N/A')} years")
    print(f"   Training Time: {results.get('time', 'N/A')} min")
else:
    print("\n📊 Check your files in:")
    print(f"   /content/drive/MyDrive/SmartRetailAI_Models/")

print("\n✅ ALL DONE!")
print("=" * 60)
```

---

## 📋 COMPLETE WORKFLOW

### **RUN THEO THỨ TỰ**:

1. ✅ **Cell 1**: Check GPU
2. ✅ **Cell 2**: Mount Drive  
3. ✅ **Cell 3**: Clone repo
4. ⏳ **Cell 4**: Setup Kaggle (upload kaggle.json)
5. ⏳ **Cell 5**: Download datasets (10-15 min)
6. ✅ **Cell 6**: Install dependencies
7. ⏳ **Cell 7**: Training (45-60 min) ← **MAIN**
8. ✅ **Cell 8**: Save to Drive
9. ✅ **Cell 9**: Summary

**Total time**: ~1 giờ 15 phút

---

## 🎯 HOẶC ALL-IN-ONE

Nếu muốn 1 cell duy nhất (không khuyến nghị):

```python
# ALL-IN-ONE CELL
import os, shutil
from google.colab import drive, files
from pathlib import Path

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Clone repo
if not os.path.exists('/content/repo'):
    !git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git /content/repo

# 3. Setup Kaggle
print("📤 Upload kaggle.json:")
uploaded = files.upload()
if 'kaggle.json' in uploaded:
    os.makedirs('/root/.kaggle', exist_ok=True)
    shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
    os.chmod('/root/.kaggle/kaggle.json', 600)

# 4. Download datasets
!pip install -q kagglehub albumentations tensorboard onnx

import kagglehub
fer2013 = kagglehub.dataset_download("msambare/fer2013")
utkface = kagglehub.dataset_download("jangedoo/utkface-new")

# 5. Train
%cd /content/repo/training_experiments
!python train_week2_lightweight.py --data_dir {fer2013} --epochs 50 --batch_size 64 --save_dir /content/checkpoints

# 6. Save to Drive
dst = Path("/content/drive/MyDrive/SmartRetailAI_Models")
dst.mkdir(parents=True, exist_ok=True)
for file in Path("/content/checkpoints").glob("*"):
    if file.is_file():
        shutil.copy(file, dst / file.name)

print(f"✅ Done! Models in: {dst}")
```

---

## ✅ CHECKLIST

- [ ] **GPU T4** enabled (Runtime → GPU)
- [ ] **Cell 1-3**: Setup (2 min)
- [ ] **Cell 4**: Upload kaggle.json
- [ ] **Cell 5**: Download data (10-15 min)
- [ ] **Cell 6**: Install deps (2 min)
- [ ] **Cell 7**: Training (45-60 min) ⏳
- [ ] **Cell 8-9**: Save & done (2 min)

**Total**: ~1 giờ 20 phút

---

**🎯 COPY TỪNG CELL VÀO COLAB VÀ CHẠY TUẦN TỰ!**
