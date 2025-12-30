# Training Experiments - Smart Retail Analytics

Thư mục này chứa code và scripts để training các deep learning models cho hệ thống Smart Retail Analytics.

## 📋 Tổng quan

Hệ thống sử dụng **Multi-task Learning** với EfficientNet-B0 backbone để nhận diện:
- **Gender**: Nam/Nữ (Binary Classification)
- **Age**: Tuổi (Regression)
- **Emotion**: 7 cảm xúc (Multi-class Classification)

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd training_experiments
venv_gpu\Scripts\activate  # Windows
# hoặc
source venv_gpu/bin/activate  # Linux/Mac
```

### 2. Test Pipeline

```bash
python scripts/test_pipeline.py
```

### 3. Training

#### Single Training Run:
```bash
python train_week2_lightweight.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --use_distillation
```

#### Automated Training (10 runs):
```bash
python train_10x_automated.py
```

### 4. Monitor Training

```bash
tensorboard --logdir checkpoints/logs
```

### 5. Evaluate & Optimize

```bash
# Optimize thresholds
python scripts/optimize_threshold.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/processed/utkface

# Evaluate với optimal thresholds
python scripts/evaluate_model.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/processed/utkface \
    --thresholds checkpoints/optimal_thresholds.json
```

### 6. Convert to ONNX

```bash
python scripts/convert_to_onnx.py \
    --model_path checkpoints/best_model.pth \
    --output_path models/multitask_efficientnet.onnx
```

## 📁 Cấu trúc Thư mục

```
training_experiments/
├── src/
│   ├── data/              # Preprocessing & Dataset
│   └── models/           # Model Architecture
├── scripts/              # Utility Scripts
├── data/                 # Datasets (raw & processed)
├── checkpoints/          # Trained Models
├── models/               # ONNX Models
└── venv_gpu/             # GPU Environment
```

Xem chi tiết: [`STRUCTURE.md`](STRUCTURE.md)

## 📊 Datasets

- **UTKFace**: Age & Gender classification
- **FER2013**: Emotion recognition
- **All Age Face Dataset**: Bổ sung dữ liệu

Xem chi tiết: [`DATASETS_INFO.md`](DATASETS_INFO.md)

## 🎯 Metrics Mục tiêu

- ✅ **Gender Accuracy**: > 94%
- ✅ **Age MAE**: < 4.0 years
- ✅ **Emotion Accuracy**: > 78%

## 📚 Tài liệu

- **[AUTO_TRAINING_GUIDE.md](AUTO_TRAINING_GUIDE.md)** - Hướng dẫn training chi tiết (BẮT ĐẦU TỪ ĐÂY!)
- **[DATASETS_INFO.md](DATASETS_INFO.md)** - Thông tin về datasets

## 🔧 Troubleshooting

### Out of Memory
- Giảm `batch_size` xuống 16 hoặc 8
- Giảm `num_workers` xuống 2

### Training quá chậm
- Kiểm tra GPU: `nvidia-smi`
- Tăng `num_workers` lên 8

### Model không converge
- Giảm learning rate: `--lr 1e-4`
- Kiểm tra dữ liệu có đúng format không

## ✅ Features

- ✅ Multi-task Learning Architecture
- ✅ Advanced LR Schedulers (CosineAnnealing + ReduceLROnPlateau)
- ✅ Dynamic Loss Weight Adjustment
- ✅ Threshold Optimization
- ✅ Early Stopping
- ✅ TensorBoard Logging
- ✅ ONNX Export

## 🚀 Next Steps

Sau khi training xong:
1. Copy model vào `ai_edge_app/models/`
2. Bắt đầu Giai đoạn 2: Edge Client Application
3. Tích hợp RetinaFace và DeepSORT
