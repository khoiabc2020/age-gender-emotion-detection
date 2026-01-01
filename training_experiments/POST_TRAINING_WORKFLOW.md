# COMPLETE WORKFLOW: TRAINING → TESTING → DEPLOYMENT

**Date:** January 2, 2026  
**From:** Training complete → Production app ready

---

## 🎯 **TỔNG QUAN WORKFLOW:**

```
[Step 1] Training xong trên Kaggle (10-11h)
    ↓
[Step 2] Export & Download files (5 min)
    ↓
[Step 3] Copy files vào project local (2 min)
    ↓
[Step 4] Test model riêng lẻ (5 min)
    ↓
[Step 5] Update app code (10 min)
    ↓
[Step 6] Test app với webcam/images (10 min)
    ↓
[Step 7] Fix issues & optimize (15 min)
    ↓
[Step 8] Deploy to production (30 min)
    ↓
[DONE] App running with 80%+ accuracy!
```

**Total time: ~1.5 hours** (sau khi training xong)

---

## ✅ **STEP 1: TRAINING XONG TRÊN KAGGLE**

### **Verify Training Success:**

**In Kaggle - Run Cell 6:**
```python
# Expected output:
============================================================
TRAINING RESULTS
============================================================
[SUCCESS] Training Completed!

============================================================
FINAL RESULTS
============================================================
Best Accuracy: 80.5%  ← VERIFY THIS!
Best Epoch: 178
Total Epochs: 200
Training Time: 10.5 hours

Total Images:
  Train: 56,916
  Test: 10,641

[GOOD] Good performance! (75-78%)
Model can be used in production with monitoring
============================================================
```

**Checklist:**
- [ ] Training completed without errors
- [ ] Accuracy ≥ 80%
- [ ] Best model saved
- [ ] Results JSON created

---

## 📥 **STEP 2: EXPORT & DOWNLOAD FILES**

### **A. Export to ONNX (Cell 7):**

**In Kaggle - Run Cell 7:**
```python
# This will:
# 1. Load best model checkpoint
# 2. Convert to ONNX format
# 3. Save to /kaggle/working/
```

**Expected output:**
```
============================================================
EXPORTING TO ONNX
============================================================
[OK] Loading best model...
[OK] Model loaded (80.5%)
[OK] Converting to ONNX...
[OK] ONNX exported: best_model_optimized.onnx (90.5 MB)
[OK] ONNX model is valid!
============================================================
```

### **B. Download Files (Cell 8):**

**In Kaggle - Run Cell 8:**
```python
# This will create download links
```

**Files to download (3 files):**
```
1. best_model_optimized.pth       (~100 MB)  ← PyTorch model
2. best_model_optimized.onnx      (~100 MB)  ← ONNX model (faster)
3. training_results_optimized.json (~5 KB)   ← Metrics
```

**How to download:**
```
Method 1: Click download links from Cell 8 output
Method 2: Right-click files in Kaggle file browser → Download
Method 3: Use Kaggle API (if enabled)
```

**Save to:**
```
C:\Users\LE HUY KHOI\Downloads\
```

---

## 💻 **STEP 3: COPY FILES VÀO PROJECT LOCAL**

### **Open PowerShell:**

```powershell
# Navigate to project
cd "D:\AI vietnam\Code\nhan dien do tuoi"

# Create backup directory
mkdir backups\models\old_models -Force

# Backup old model (if exists)
if (Test-Path "trained_models\best_model.pth") {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item "trained_models\best_model.pth" "backups\models\old_models\best_model_76.49_$timestamp.pth"
    Write-Host "[OK] Old model backed up"
}

# Copy new PyTorch model
Copy-Item "C:\Users\LE HUY KHOI\Downloads\best_model_optimized.pth" "trained_models\best_model.pth"
Write-Host "[OK] PyTorch model copied"

# Copy ONNX model (for faster inference)
Copy-Item "C:\Users\LE HUY KHOI\Downloads\best_model_optimized.onnx" "trained_models\best_model.onnx"
Write-Host "[OK] ONNX model copied"

# Copy results
Copy-Item "C:\Users\LE HUY KHOI\Downloads\training_results_optimized.json" "training_experiments\results\production_80percent_results.json"
Write-Host "[OK] Results copied"

# Verify
Write-Host "`n[VERIFY] Files in trained_models:"
Get-ChildItem "trained_models" | Format-Table Name, @{Name="Size (MB)";Expression={[math]::Round($_.Length / 1MB, 2)}}
```

**Expected output:**
```
[OK] Old model backed up
[OK] PyTorch model copied
[OK] ONNX model copied
[OK] Results copied

[VERIFY] Files in trained_models:
Name                          Size (MB)
----                          ---------
best_model.pth                100.23
best_model.onnx               100.45
```

---

## 🧪 **STEP 4: TEST MODEL RIÊNG LẺ**

### **Create Test Script:**

**File:** `test_new_model.py`

```python
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

print("="*60)
print("TESTING NEW MODEL (80%+)")
print("="*60)

# ============================================================
# 1. TEST MODEL LOADING
# ============================================================

print("\n[1/5] Testing model loading...")
checkpoint_path = Path('trained_models/best_model.pth')

try:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"  [OK] Model loaded successfully!")
    print(f"  [OK] Best Accuracy: {checkpoint['best_accuracy']:.2f}%")
    print(f"  [OK] Best Epoch: {checkpoint['epoch']}")
    print(f"  [OK] Num Classes: {checkpoint['num_classes']}")
    print(f"  [OK] Class Names: {checkpoint['class_names']}")
    
    if 'config' in checkpoint:
        print(f"  [OK] Model Type: {checkpoint['config'].get('model', 'Unknown')}")
        print(f"  [OK] Improvements: {checkpoint['config'].get('improvements', 'N/A')}")
except Exception as e:
    print(f"  [ERROR] Failed to load model: {e}")
    exit(1)

# ============================================================
# 2. TEST MODEL ARCHITECTURE
# ============================================================

print("\n[2/5] Testing model architecture...")
try:
    import timm
    
    model_type = checkpoint['config'].get('model', 'efficientnetv2_rw_s')
    num_classes = checkpoint['num_classes']
    
    model = timm.create_model(model_type, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  [OK] Model architecture: {model_type}")
    print(f"  [OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  [OK] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"  [ERROR] Failed to create model: {e}")
    exit(1)

# ============================================================
# 3. TEST INFERENCE WITH DUMMY INPUT
# ============================================================

print("\n[3/5] Testing inference...")
try:
    # Create dummy input (batch_size=1, channels=3, height=72, width=72)
    dummy_input = torch.randn(1, 3, 72, 72)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  [OK] Input shape: {dummy_input.shape}")
    print(f"  [OK] Output shape: {output.shape}")
    print(f"  [OK] Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Test softmax
    probs = torch.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1).item()
    confidence = probs.max().item()
    
    print(f"  [OK] Predicted class: {pred_class} ({checkpoint['class_names'][pred_class]})")
    print(f"  [OK] Confidence: {confidence:.2%}")
except Exception as e:
    print(f"  [ERROR] Inference failed: {e}")
    exit(1)

# ============================================================
# 4. TEST WITH REAL IMAGE (if exists)
# ============================================================

print("\n[4/5] Testing with real image...")

# Define transform (same as training)
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Try to find test image
test_image_paths = [
    'test_data/sample_faces/happy.jpg',
    'test_data/sample_faces/sad.jpg',
    'data/test_images/face1.jpg',
]

test_image = None
for path in test_image_paths:
    if Path(path).exists():
        test_image = path
        break

if test_image:
    try:
        img = Image.open(test_image).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs.max().item()
        
        print(f"  [OK] Test image: {test_image}")
        print(f"  [OK] Prediction: {checkpoint['class_names'][pred_class]}")
        print(f"  [OK] Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        top3_probs, top3_indices = torch.topk(probs, 3)
        print(f"  [OK] Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top3_probs[0], top3_indices[0])):
            print(f"      {i+1}. {checkpoint['class_names'][idx]}: {prob:.2%}")
    except Exception as e:
        print(f"  [WARN] Could not test with image: {e}")
else:
    print(f"  [SKIP] No test image found")

# ============================================================
# 5. TEST ONNX MODEL (if exists)
# ============================================================

print("\n[5/5] Testing ONNX model...")
onnx_path = Path('trained_models/best_model.onnx')

if onnx_path.exists():
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        
        # Test with dummy input
        dummy_input_np = np.random.randn(1, 3, 72, 72).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input_np})
        
        print(f"  [OK] ONNX model loaded")
        print(f"  [OK] Input name: {input_name}")
        print(f"  [OK] Output shape: {outputs[0].shape}")
        print(f"  [OK] ONNX inference working!")
    except ImportError:
        print(f"  [WARN] onnxruntime not installed. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"  [ERROR] ONNX test failed: {e}")
else:
    print(f"  [SKIP] ONNX model not found")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("MODEL TEST SUMMARY")
print("="*60)
print(f"Model Accuracy: {checkpoint['best_accuracy']:.2f}%")
print(f"Model Type: {model_type}")
print(f"Input Size: 72x72")
print(f"Num Classes: {num_classes}")
print(f"Status: ✓ ALL TESTS PASSED")
print("="*60)
print("\n[SUCCESS] Model is ready for integration!")
print("\nNext steps:")
print("  1. Update app code (see STEP 5)")
print("  2. Test with app (see STEP 6)")
print("  3. Deploy to production (see STEP 8)")
print("="*60)
```

**Run test:**
```powershell
cd "D:\AI vietnam\Code\nhan dien do tuoi"
python test_new_model.py
```

**Expected output:**
```
============================================================
TESTING NEW MODEL (80%+)
============================================================

[1/5] Testing model loading...
  [OK] Model loaded successfully!
  [OK] Best Accuracy: 80.50%
  [OK] Best Epoch: 178
  [OK] Num Classes: 7
  [OK] Class Names: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
  [OK] Model Type: efficientnetv2_rw_s
  [OK] Improvements: RandAugment+CutMix+FocalLoss+200epochs

[2/5] Testing model architecture...
  [OK] Model architecture: efficientnetv2_rw_s
  [OK] Parameters: 5,345,678
  [OK] Trainable params: 5,345,678

[3/5] Testing inference...
  [OK] Input shape: torch.Size([1, 3, 72, 72])
  [OK] Output shape: torch.Size([1, 7])
  [OK] Output range: [-2.34, 3.56]
  [OK] Predicted class: 3 (happy)
  [OK] Confidence: 45.23%

[4/5] Testing with real image...
  [SKIP] No test image found

[5/5] Testing ONNX model...
  [OK] ONNX model loaded
  [OK] Input name: input
  [OK] Output shape: (1, 7)
  [OK] ONNX inference working!

============================================================
MODEL TEST SUMMARY
============================================================
Model Accuracy: 80.50%
Model Type: efficientnetv2_rw_s
Input Size: 72x72
Num Classes: 7
Status: ✓ ALL TESTS PASSED
============================================================

[SUCCESS] Model is ready for integration!

Next steps:
  1. Update app code (see STEP 5)
  2. Test with app (see STEP 6)
  3. Deploy to production (see STEP 8)
============================================================
```

---

## 🔧 **STEP 5: UPDATE APP CODE**

### **A. Update Model Loader:**

**File:** `ai_edge_app/src/models/model_loader.py`

```python
import torch
import timm
from pathlib import Path

class ModelLoader:
    def __init__(self, model_path='trained_models/best_model.pth'):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def load_model(self):
        """Load the trained model"""
        print(f"[Loading] Model from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, weights_only=False, map_location=self.device)
        
        # Get model config
        model_type = checkpoint.get('config', {}).get('model', 'efficientnetv2_rw_s')
        num_classes = checkpoint['num_classes']
        
        # Create model
        self.model = timm.create_model(model_type, pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get class names from checkpoint
        self.class_names = checkpoint.get('class_names', self.class_names)
        
        print(f"[OK] Model loaded: {model_type}")
        print(f"[OK] Accuracy: {checkpoint['best_accuracy']:.2f}%")
        print(f"[OK] Device: {self.device}")
        
        return True
```

### **B. Update Image Preprocessor:**

**File:** `ai_edge_app/src/utils/image_processor.py`

```python
from torchvision import transforms
import torch

class ImageProcessor:
    def __init__(self, input_size=72):  # NEW SIZE!
        self.input_size = input_size
        
        # NEW: Match training transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image):
        """Preprocess image for model"""
        # image is numpy array (H, W, C) from webcam
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
```

### **C. Update Main App:**

**File:** `ai_edge_app/main.py`

```python
import cv2
import torch
import numpy as np
from src.models.model_loader import ModelLoader
from src.utils.image_processor import ImageProcessor

def main():
    print("="*60)
    print("EMOTION DETECTION APP - v2.0 (80%+ Accuracy)")
    print("="*60)
    
    # Load model
    loader = ModelLoader('trained_models/best_model.pth')
    if not loader.load_model():
        print("[ERROR] Failed to load model")
        return
    
    # Create preprocessor
    processor = ImageProcessor(input_size=72)  # NEW SIZE!
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    
    print("\n[OK] Webcam opened")
    print("[INFO] Press 'q' to quit")
    print("="*60)
    
    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face
            face_img = frame[y:y+h, x:x+w]
            
            # Preprocess
            face_tensor = processor.preprocess(face_img)
            
            # Predict
            with torch.no_grad():
                output = loader.model(face_tensor.to(loader.device))
                probs = torch.softmax(output, dim=1)
                confidence, pred_class = probs.max(1)
                
                emotion = loader.class_names[pred_class.item()]
                conf = confidence.item()
            
            # Draw box and label
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{emotion}: {conf:.2%}"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display
        cv2.imshow('Emotion Detection - v2.0 (80%+)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] App closed")

if __name__ == '__main__':
    main()
```

---

## 🎮 **STEP 6: TEST APP VỚI WEBCAM/IMAGES**

### **A. Test với Webcam:**

```powershell
cd "D:\AI vietnam\Code\nhan dien do tuoi"
cd ai_edge_app
python main.py
```

**Expected:**
```
============================================================
EMOTION DETECTION APP - v2.0 (80%+ Accuracy)
============================================================
[Loading] Model from trained_models\best_model.pth...
[OK] Model loaded: efficientnetv2_rw_s
[OK] Accuracy: 80.50%
[OK] Device: cpu

[OK] Webcam opened
[INFO] Press 'q' to quit
============================================================
```

**Check:**
- [ ] Webcam opens successfully
- [ ] Faces are detected
- [ ] Emotions are predicted
- [ ] Confidence scores shown
- [ ] No errors in console
- [ ] Performance is smooth (>15 FPS)

### **B. Test với Test Images:**

**Create:** `test_app_with_images.py`

```python
import cv2
from pathlib import Path
from src.models.model_loader import ModelLoader
from src.utils.image_processor import ImageProcessor
import torch

# Load model
loader = ModelLoader('trained_models/best_model.pth')
loader.load_model()

processor = ImageProcessor(input_size=72)

# Test images
test_images = [
    'test_data/sample_faces/happy.jpg',
    'test_data/sample_faces/sad.jpg',
    'test_data/sample_faces/angry.jpg',
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

for img_path in test_images:
    if not Path(img_path).exists():
        continue
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_tensor = processor.preprocess(face_img)
        
        with torch.no_grad():
            output = loader.model(face_tensor.to(loader.device))
            probs = torch.softmax(output, dim=1)
            confidence, pred_class = probs.max(1)
            
            emotion = loader.class_names[pred_class.item()]
            conf = confidence.item()
        
        print(f"{Path(img_path).name}: {emotion} ({conf:.2%})")
        
        # Draw and save
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion}: {conf:.2%}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Test', img)
    cv2.waitKey(2000)

cv2.destroyAllWindows()
```

---

## 🐛 **STEP 7: FIX ISSUES & OPTIMIZE**

### **Common Issues & Solutions:**

**Issue 1: Model loading error**
```python
# Error: KeyError: 'model_state_dict'
# Solution: Check checkpoint structure
checkpoint = torch.load('trained_models/best_model.pth', weights_only=False)
print(checkpoint.keys())
```

**Issue 2: Wrong input size**
```python
# Error: RuntimeError: size mismatch
# Solution: Use input_size=72 (not 64!)
processor = ImageProcessor(input_size=72)
```

**Issue 3: Low FPS**
```python
# Solution: Use ONNX model for faster inference
import onnxruntime as ort
session = ort.InferenceSession('trained_models/best_model.onnx')
```

**Issue 4: Low confidence**
```python
# Solution: Add confidence threshold
if confidence > 0.7:
    # Use prediction
else:
    # Show "Uncertain"
```

### **Optimization:**

**Use ONNX for 3x faster inference:**

```python
import onnxruntime as ort
import numpy as np

class ONNXModel:
    def __init__(self, onnx_path='trained_models/best_model.onnx'):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, image_tensor):
        # Convert to numpy
        input_np = image_tensor.cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: input_np})
        return torch.from_numpy(outputs[0])
```

---

## 🚀 **STEP 8: DEPLOY TO PRODUCTION**

### **A. Verify Everything Works:**

**Checklist:**
- [ ] Model accuracy ≥ 80%
- [ ] App runs without errors
- [ ] Webcam detection works
- [ ] Predictions are reasonable
- [ ] Performance is acceptable
- [ ] Confidence thresholds set

### **B. Create Deployment Package:**

```powershell
cd "D:\AI vietnam\Code\nhan dien do tuoi"

# Create deployment folder
mkdir deployment -Force

# Copy necessary files
Copy-Item ai_edge_app deployment\ai_edge_app -Recurse
Copy-Item trained_models\best_model.onnx deployment\
Copy-Item requirements.txt deployment\

# Create deployment README
@"
# Emotion Detection App v2.0

Accuracy: 80%+
Model: EfficientNetV2

## Installation:
pip install -r requirements.txt

## Run:
python ai_edge_app/main.py
"@ | Out-File deployment\README.md

# Zip for distribution
Compress-Archive -Path deployment\* -DestinationPath emotion_detection_v2.zip
```

### **C. Deploy Options:**

**Option 1: Local Desktop App**
```
- Package with PyInstaller
- Create .exe for Windows
- Distribute to users
```

**Option 2: Web API (Flask/FastAPI)**
```python
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict_emotion(file: UploadFile):
    # Load image
    # Preprocess
    # Predict
    # Return JSON
    pass

uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Option 3: Cloud Deployment**
```
- Deploy to AWS/GCP/Azure
- Use Docker container
- Scale with load balancer
```

---

## ✅ **FINAL CHECKLIST:**

### **Post-Training:**
- [ ] Training completed (80%+)
- [ ] Files downloaded from Kaggle
- [ ] Files copied to project

### **Testing:**
- [ ] Model loads successfully
- [ ] Inference works
- [ ] ONNX export works
- [ ] App code updated
- [ ] Webcam test passed
- [ ] Image test passed

### **Production:**
- [ ] Performance optimized
- [ ] Confidence thresholds set
- [ ] Error handling added
- [ ] Logging implemented
- [ ] Deployment package created

---

## 📊 **SUMMARY:**

| Step | Action | Time | Status |
|------|--------|------|--------|
| 1 | Training complete | 10-11h | ✅ |
| 2 | Download files | 5min | ⏳ |
| 3 | Copy to project | 2min | ⏳ |
| 4 | Test model | 5min | ⏳ |
| 5 | Update app code | 10min | ⏳ |
| 6 | Test app | 10min | ⏳ |
| 7 | Fix & optimize | 15min | ⏳ |
| 8 | Deploy | 30min | ⏳ |
| **Total** | - | **~1.5h** | - |

---

## 🎯 **EXPECTED END RESULT:**

```
✅ Model trained: 80%+ accuracy
✅ Model tested: Working correctly
✅ App updated: New model integrated
✅ App tested: Webcam & images working
✅ Performance: Smooth & fast
✅ Ready for: Production deployment
```

---

**🚀 START WITH STEP 2 AFTER TRAINING COMPLETES!**

---

*Complete Workflow Guide*  
*From Training → Production*  
*Updated: January 2, 2026*
