# ⚡ QUICK START - TRAIN ON COLAB

**Train model với GPU miễn phí trong 5 bước - 5 phút setup!**

---

## 📋 CHECKLIST (2 phút)

### 1. Kaggle API Key ✅

**Lấy token**:
1. Vào: https://www.kaggle.com/settings/account
2. Scroll xuống section "API"
3. Click **"Create New API Token"**
4. Download file `kaggle.json`
5. **Giữ file này** - sẽ upload lên Colab

### 2. GitHub Repo ✅

**Repository đã public**: https://github.com/khoiabc2020/age-gender-emotion-detection

✅ Code đã được push (vừa xong)

---

## 🚀 5 BƯỚC - BẮT ĐẦU TRAINING

### BƯỚC 1: Mở Colab Notebook (30s)

**Click link này**:
👉 https://colab.research.google.com/github/khoiabc2020/age-gender-emotion-detection/blob/main/training_experiments/notebooks/train_on_colab_auto.ipynb

Hoặc:
1. Vào: https://colab.research.google.com/
2. File → Open notebook → GitHub tab
3. Nhập URL: `khoiabc2020/age-gender-emotion-detection`
4. Chọn: `training_experiments/notebooks/train_on_colab_auto.ipynb`

### BƯỚC 2: Enable GPU (30s)

**QUAN TRỌNG!**

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator**: Chọn **GPU**
3. **GPU type**: **T4** (free)
4. Click **Save**

### BƯỚC 3: Run All Cells (30s)

**Cách nhanh nhất**:
- **Runtime** → **Run all** (hoặc **Ctrl+F9**)

Notebook sẽ tự động:
- ✅ Check GPU
- ✅ Install dependencies
- ✅ Mount Google Drive
- ✅ Clone code từ GitHub
- ✅ Setup datasets (cần upload kaggle.json - bước 4)

### BƯỚC 4: Upload Kaggle Token (1 phút)

Khi notebook chạy đến cell "Setup Kaggle API", sẽ có popup yêu cầu upload file.

**Upload file `kaggle.json`** đã download ở bước chuẩn bị.

### BƯỚC 5: Chờ Training Xong (1 giờ)

Training sẽ tự động chạy và hiển thị progress:

```
Epoch 1/50
Training: 100%|██████████| 215/215 [02:30<00:00]
Loss: 2.45, Gender Acc: 85.3%, Emotion Acc: 68.2%

Epoch 2/50
Training: 100%|██████████| 215/215 [02:28<00:00]
Loss: 2.12, Gender Acc: 87.5%, Emotion Acc: 71.4%
...
```

**Thời gian**: ~45-60 phút với GPU T4

---

## 📥 LẤY KẾT QUẢ (2 phút)

### Models Tự Động Lưu Trong Google Drive

```
Google Drive/
└── MyDrive/
    └── age_gender_emotion_training/
        ├── best_model.pth       # PyTorch model
        ├── model.onnx           # ONNX model (cho edge app)
        ├── training_results.json # Metrics
        └── logs/                # TensorBoard logs
```

### Download Về Máy

**Option 1**: Từ Google Drive
- Mở Google Drive
- Navigate đến folder `age_gender_emotion_training`
- Download files

**Option 2**: Từ Colab
- Cell cuối cùng sẽ tự động zip và download

---

## ✅ SAU KHI TRAINING XONG

### Copy Model vào Edge App

```bash
# Trên Windows
cd "D:\AI vietnam\Code\nhan dien do tuoi"

# Copy ONNX model
copy Downloads\model.onnx ai_edge_app\models\mobileone_multitask.onnx
```

### Test Model

```bash
cd ai_edge_app
python main.py
```

### Commit Kết Quả

```bash
git add .
git commit -m "Add trained model from Colab"
git push
```

---

## 🎯 KẾT QUẢ MONG ĐỢI

| Metric | Target | Actual (Example) |
|--------|--------|------------------|
| Gender Accuracy | > 90% | 92.5% ✅ |
| Emotion Accuracy | > 75% | 78.3% ✅ |
| Age MAE | < 4.0 years | 3.8 years ✅ |
| Training Time | ~1 hour | 56 minutes ✅ |
| Model Size | ~25MB | 24.8MB ✅ |

---

## 🔧 SETTINGS (Tuỳ Chỉnh)

### Thay Đổi Cấu Hình Training

Trong notebook, tìm cell "Cấu hình Training":

```python
# ========================================
# TRAINING CONFIG - SỬA TẠI ĐÂY
# ========================================

EPOCHS = 50              # Số epochs (càng nhiều càng tốt)
BATCH_SIZE = 64          # Batch size (GPU T4: 64-128)
LEARNING_RATE = 0.001    # Learning rate
USE_DISTILLATION = True  # Knowledge distillation (tăng accuracy)
USE_QAT = False          # Quantization (chậm hơn 30%)
```

### Recommendations

**For Best Accuracy**:
```python
EPOCHS = 100
BATCH_SIZE = 64
USE_DISTILLATION = True
USE_QAT = False  # Chạy sau nếu cần
```

**For Quick Test**:
```python
EPOCHS = 20
BATCH_SIZE = 128
USE_DISTILLATION = False
USE_QAT = False
```

---

## 💡 TIPS

### Tăng Tốc

- ✅ Batch size lớn: `BATCH_SIZE = 128` (nếu GPU đủ memory)
- ✅ Tắt QAT: `USE_QAT = False` (nhanh hơn 30%)
- ✅ Giảm epochs: `EPOCHS = 30` (nhanh hơn nhưng accuracy thấp)

### Tăng Accuracy

- ✅ Epochs nhiều: `EPOCHS = 100`
- ✅ Enable distillation: `USE_DISTILLATION = True`
- ✅ Learning rate scheduler (đã có trong script)

### Tiết Kiệm GPU Time

- 📱 **Colab app**: Install app để nhận notification khi done
- 🌙 **Overnight training**: Chạy trước khi đi ngủ
- 💰 **Colab Pro**: $10/month, GPU V100 nhanh hơn 2x

---

## 🆘 TROUBLESHOOTING

### Lỗi "No GPU Available"

**Fix**:
1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. Save → Reconnect

### Lỗi Kaggle API

**Fix**:
- Verify file `kaggle.json` đúng format
- Re-upload file
- Check: https://www.kaggle.com/settings/account

### Out of Memory

**Fix**:
```python
BATCH_SIZE = 32  # Giảm batch size
```

### Colab Disconnect

**No problem!**
- Models đã lưu trong Google Drive
- Reconnect và chạy tiếp từ cell training

---

## 📊 SO SÁNH: LOCAL vs COLAB

| Aspect | Local CPU | Colab GPU T4 |
|--------|-----------|--------------|
| **Setup Time** | 5 min | 5 min |
| **Training Time** | 6-8 hours | ~1 hour |
| **Speed** | 1x | **8x faster** |
| **Cost** | Free | **Free** |
| **Convenience** | Must keep PC on | Can close browser |
| **GPU Memory** | 0 | 15GB |

**Verdict**: ✅ **COLAB WINS!**

---

## 🎉 TÓM TẮT

### Thời Gian Tổng

- ⏱️ Setup: **5 phút**
- ⏱️ Training: **1 giờ** (GPU T4)
- ⏱️ Download: **2 phút**
- **TOTAL**: ~**1 giờ 10 phút**

### So với Local CPU

- 💻 Local CPU: **6-8 giờ**
- ☁️ Colab GPU: **~1 giờ**
- ⚡ **Nhanh hơn 8x**

### Chi Phí

- 💰 **$0** - Hoàn toàn miễn phí!

---

## 🚀 BẮT ĐẦU NGAY!

**Step 1**: Lấy Kaggle token
👉 https://www.kaggle.com/settings/account

**Step 2**: Mở Colab notebook
👉 https://colab.research.google.com/github/khoiabc2020/age-gender-emotion-detection/blob/main/training_experiments/notebooks/train_on_colab_auto.ipynb

**Step 3**: Runtime → GPU → Run all → Upload kaggle.json

**Step 4**: Đợi 1 giờ → Download models → Done!

---

## 📚 TÀI LIỆU ĐẦY ĐỦ

- **Chi tiết**: [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)
- **GitHub Repo**: https://github.com/khoiabc2020/age-gender-emotion-detection
- **Colab Docs**: https://colab.research.google.com/

---

**⚡ Training với GPU miễn phí - Nhanh gấp 8 lần!**

**Last Updated**: 2025-12-31
