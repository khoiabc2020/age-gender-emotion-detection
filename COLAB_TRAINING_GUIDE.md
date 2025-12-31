# 🚀 HƯỚNG DẪN TRAIN TRÊN GOOGLE COLAB

**Train model với GPU miễn phí - Nhanh hơn CPU 10-20 lần!**

---

## 📋 CHUẨN BỊ

### 1. Code đã lên GitHub ✅
- Repository: https://github.com/khoiabc2020/age-gender-emotion-detection
- Code đã được push (bạn vừa làm xong)

### 2. Kaggle API Key
**Lấy Kaggle API key**:
1. Truy cập: https://www.kaggle.com/settings/account
2. Scroll xuống **"API"** section
3. Click **"Create New API Token"**
4. File `kaggle.json` sẽ được download
5. **GIỮ FILE NÀY** - sẽ cần upload lên Colab

### 3. Google Account
- Có tài khoản Google (Gmail)
- Truy cập được Google Colab

---

## 🚀 BƯỚC 1: MỞ COLAB NOTEBOOK

### Option 1: Upload Notebook (Khuyến nghị)

1. **Download notebook** từ repo:
   - File: `training_experiments/notebooks/TRAIN_ON_COLAB.ipynb`
   - Hoặc: https://github.com/khoiabc2020/age-gender-emotion-detection/blob/main/training_experiments/notebooks/TRAIN_ON_COLAB.ipynb

2. **Truy cập Colab**:
   - https://colab.research.google.com/

3. **Upload notebook**:
   - File → Upload notebook
   - Chọn file `TRAIN_ON_COLAB.ipynb`

### Option 2: Tạo Notebook Mới

1. Truy cập: https://colab.research.google.com/
2. File → New notebook
3. Copy code từ `TRAIN_ON_COLAB.ipynb`

---

## ⚙️ BƯỚC 2: CHỌN GPU

**QUAN TRỌNG!** Phải enable GPU:

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator**: Chọn **GPU**
3. **GPU type**: **T4** (free) hoặc **V100** (Colab Pro)
4. **Save**

**Verify GPU**:
```python
!nvidia-smi
```

Kết quả sẽ hiển thị:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   50C    P0    27W /  70W |      0MiB / 15360MiB |      0%      Default |
```

---

## 🏃 BƯỚC 3: CHẠY TRAINING

### Chạy Tất Cả Cells

**Cách nhanh nhất**:
- **Runtime** → **Run all** (Ctrl+F9)

Notebook sẽ tự động:
1. ✅ Kiểm tra GPU
2. ✅ Mount Google Drive
3. ✅ Clone code từ GitHub
4. ✅ Cài đặt dependencies
5. ✅ Setup Kaggle API (cần upload `kaggle.json`)
6. ✅ Download datasets
7. ✅ Chạy training
8. ✅ Lưu models về Drive
9. ✅ Export ONNX

### Hoặc Chạy Từng Cell

**Cell 1-7**: Setup
- Chạy lần lượt từ cell 1 → 7
- **Cell 5**: Upload `kaggle.json` khi được yêu cầu

**Cell 8**: Training (Quan trọng!)
```python
# CẤU HÌNH TRAINING - SỬA TẠI ĐÂY
EPOCHS = 50              # Số epochs
BATCH_SIZE = 64          # Batch size (GPU xử lý được lớn)
LEARNING_RATE = 0.001    # Learning rate
USE_DISTILLATION = True  # Knowledge distillation
USE_QAT = False          # Quantization (chậm hơn)
```

**Chạy training**:
- Click vào cell 8
- Shift+Enter để chạy

**Cell 9-12**: Lưu kết quả
- Tự động chạy sau training

---

## ⏱️ THỜI GIAN TRAINING

### GPU T4 (Free)
| Epochs | Batch Size | Time |
|--------|-----------|------|
| 30 | 64 | ~30 phút |
| 50 | 64 | ~45-60 phút |
| 100 | 64 | ~2 giờ |

### So Sánh CPU vs GPU
| Device | 50 Epochs | Speed |
|--------|-----------|-------|
| **CPU** | 6-8 giờ | 1x |
| **GPU T4** | ~1 giờ | **8x nhanh hơn** |
| **GPU V100** | ~30 phút | **15x nhanh hơn** |

---

## 📊 MONITOR TRAINING

### Xem Progress

Training sẽ hiển thị:
```
Epoch 1/50
Training: 100%|██████████| 215/215 [02:30<00:00, 2.05s/it]
Validation: 100%|██████████| 54/54 [00:30<00:00, 1.80s/it]

Epoch 1 - Loss: 2.45, Gender Acc: 85.3%, Emotion Acc: 68.2%, Age MAE: 5.2
```

### TensorBoard (Real-time)

Trong cell "View Training Results":
```python
%load_ext tensorboard
%tensorboard --logdir /content/checkpoints/colab_training/logs
```

Sẽ hiển thị charts real-time:
- Loss curves
- Accuracy metrics
- Learning rate

---

## 💾 LƯU KẾT QUẢ

### Auto Save to Google Drive

Notebook tự động lưu:
```
Google Drive/
└── MyDrive/
    └── SmartRetailAI/
        └── models/
            └── colab_training/
                ├── best_model.pth
                ├── model.onnx
                ├── training_results.json
                └── logs/
```

### Download Về Máy

**Option 1**: Download từ Google Drive
- Mở Google Drive
- Navigate đến folder trên
- Download files

**Option 2**: Download trực tiếp từ Colab
```python
# Cell cuối cùng
from google.colab import files
files.download('/content/trained_models.zip')
```

---

## 🎯 KẾT QUẢ MONG ĐỢI

### Metrics Target

| Metric | Target | Notes |
|--------|--------|-------|
| **Gender Accuracy** | > 90% | Binary classification |
| **Emotion Accuracy** | > 75% | 6 classes |
| **Age MAE** | < 4.0 years | Regression |
| **Model Size** | ~25MB | ONNX format |

### Example Output

```json
{
  "best_epoch": 35,
  "gender_accuracy": 92.5,
  "emotion_accuracy": 78.3,
  "age_mae": 3.8,
  "total_time": "56 minutes"
}
```

---

## 🔄 NẾU TRAINING BỊ DISCONNECT

Colab có thể disconnect sau 12 giờ idle. Nếu bị disconnect:

### Models đã được lưu!

1. **Check Google Drive**:
   - Models đã được save trong Drive
   - Không mất tiến độ

2. **Resume Training**:
   ```python
   # Trong cell training, thêm:
   --resume_from /content/drive/MyDrive/SmartRetailAI/models/colab_training/last_checkpoint.pth
   ```

3. **Reconnect & Run**:
   - Runtime → Reconnect
   - Chạy lại từ cell 8 (Training)

---

## ✅ SAU KHI TRAINING XONG

### 1. Download Models

**Files cần download**:
- ✅ `best_model.pth` - PyTorch model (25MB)
- ✅ `model.onnx` - ONNX model cho edge app (25MB)
- ✅ `training_results.json` - Metrics

### 2. Copy Model vào Edge App

```bash
# Trên máy local
cd "D:\AI vietnam\Code\nhan dien do tuoi"

# Copy ONNX model
copy Downloads\model.onnx ai_edge_app\models\mobileone_multitask.onnx
```

### 3. Test Model

```bash
# Test edge app
cd ai_edge_app
python main.py
```

### 4. Update Documentation

```bash
# Update TRAINING_RESULTS.md với metrics mới
# Commit và push lên GitHub
git add .
git commit -m "Training completed on Colab - Add model results"
git push
```

---

## 💡 TIPS & TRICKS

### Tăng Tốc Training

1. **Batch Size Lớn Hơn**:
   ```python
   BATCH_SIZE = 128  # GPU T4 có thể xử lý
   ```

2. **Mixed Precision**:
   ```python
   # Đã enabled mặc định trong script
   # Nhanh hơn 2x, dùng ít memory hơn
   ```

3. **Reduce Epochs**:
   ```python
   EPOCHS = 30  # Nhanh hơn, accuracy có thể thấp hơn
   ```

### Save GPU Time

- **Không chạy QAT**: `USE_QAT = False` (nhanh hơn 30%)
- **Chạy overnight**: Để Colab chạy qua đêm
- **Pro version**: $10/month, GPU V100 nhanh hơn 2x

### Debug Issues

**Lỗi "Out of Memory"**:
```python
BATCH_SIZE = 32  # Giảm batch size
```

**Lỗi "Kaggle API"**:
- Verify `kaggle.json` uploaded đúng
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Training chậm**:
- Verify GPU enabled: `!nvidia-smi`
- Check GPU usage: Nếu 0%, có vấn đề

---

## 📚 TÀI LIỆU THAM KHẢO

### Colab
- Official Docs: https://colab.research.google.com/
- GPU Guide: https://colab.research.google.com/notebooks/gpu.ipynb

### Kaggle
- API Docs: https://www.kaggle.com/docs/api
- Datasets: https://www.kaggle.com/datasets

### GitHub Repo
- Code: https://github.com/khoiabc2020/age-gender-emotion-detection
- Issues: https://github.com/khoiabc2020/age-gender-emotion-detection/issues

---

## 🆘 TROUBLESHOOTING

### Lỗi Thường Gặp

| Lỗi | Giải pháp |
|-----|-----------|
| No GPU | Runtime → Change runtime → GPU |
| Kaggle 401 | Upload đúng `kaggle.json` |
| Out of Memory | Giảm batch size |
| Disconnect | Models đã lưu trong Drive |
| Clone failed | Check GitHub repo public |

### Get Help

- GitHub Issues: https://github.com/khoiabc2020/age-gender-emotion-detection/issues
- Colab FAQ: https://research.google.com/colaboratory/faq.html

---

## ✅ CHECKLIST HOÀN THÀNH

### Trước Training
- [ ] Code đã push lên GitHub
- [ ] Có `kaggle.json`
- [ ] Colab GPU enabled
- [ ] Google Drive mounted

### Trong Training
- [ ] Training đang chạy
- [ ] Metrics improving
- [ ] GPU usage > 80%
- [ ] No errors

### Sau Training
- [ ] Models saved to Drive
- [ ] Downloaded về máy
- [ ] Copied to `ai_edge_app/models/`
- [ ] Tested edge app
- [ ] Updated documentation
- [ ] Pushed to GitHub

---

## 🎉 TÓM TẮT

### Quick Start (5 bước)

1. **Upload notebook** lên Colab
2. **Enable GPU** (Runtime → GPU)
3. **Run all cells** (Ctrl+F9)
4. **Upload kaggle.json** khi được hỏi
5. **Đợi ~1 giờ** → Done!

### Kết Quả

- ✅ Model trained với GPU (nhanh 8-10x)
- ✅ Accuracy > 85% (gender), > 75% (emotion)
- ✅ ONNX exported, ready for edge app
- ✅ All saved in Google Drive

---

**🚀 BẮT ĐẦU TRAINING NGAY!**

**Notebook**: `training_experiments/notebooks/TRAIN_ON_COLAB.ipynb`

**Colab**: https://colab.research.google.com/

**Thời gian**: ~1 giờ với GPU

**Cost**: FREE!

---

**Last Updated**: 2025-12-31
