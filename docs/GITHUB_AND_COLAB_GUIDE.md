# 📤 Hướng dẫn GitHub & Google Colab

Hướng dẫn đầy đủ về cách upload code lên GitHub và train model trên Google Colab.

## 📋 Mục lục

1. [Upload Code lên GitHub](#upload-code-lên-github)
2. [Train trên Google Colab](#train-trên-google-colab)
3. [Sync Code với GitHub](#sync-code-với-github)

---

## 📤 Upload Code lên GitHub

### ⚡ Cách Nhanh Nhất (3 bước)

#### Bước 1: Tạo Personal Access Token

GitHub **KHÔNG CÒN** chấp nhận password từ năm 2021. Bạn **PHẢI** dùng **Personal Access Token**.

1. Truy cập: **https://github.com/settings/tokens**
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Điền thông tin:
   - **Note**: "My Computer"
   - **Expiration**: "90 days" hoặc "No expiration"
   - **Select scopes**: Tích chọn **`repo`** (full control)
4. Click **"Generate token"**
5. **COPY TOKEN NGAY** (chỉ hiện 1 lần!)
   - Token có dạng: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

#### Bước 2: Chạy Script Push

```bash
# Chạy script này
scripts/push_to_github.bat
```

Khi được hỏi:
- **Username**: `khoiabc2020`
- **Password**: **PASTE TOKEN VÀO** (không phải password thật)

#### Bước 3: Xác nhận

Sau khi push thành công, xem code tại:
**https://github.com/khoiabc2020/age-gender-emotion-detection**

### 🔄 Sync Code Sau Khi Sửa

#### Cách 1: Script Tự Động (Khuyến nghị)

```bash
training_experiments/scripts/auto_sync.bat
```

#### Cách 2: Tự Động Real-time

```bash
training_experiments/scripts/watch_sync.bat
```

Script này sẽ tự động commit và push mỗi khi bạn sửa code.

---

## 🚀 Train trên Google Colab

### ⚡ Cách Nhanh Nhất

#### Bước 1: Upload Code lên GitHub

Đảm bảo code đã được push lên GitHub (xem phần trên).

#### Bước 2: Mở Colab

1. Truy cập: **https://colab.research.google.com/**
2. Upload notebook: `training_experiments/notebooks/train_on_colab_auto.ipynb`
3. **Chọn GPU**: Runtime → Change runtime type → GPU (T4)

#### Bước 3: Chạy Tự Động

- Runtime → Run all (Ctrl+F9)
- Notebook sẽ tự động:
  - ✅ Cài đặt dependencies
  - ✅ Kiểm tra GPU
  - ✅ Mount Google Drive
  - ✅ Clone code từ GitHub
  - ✅ Setup dữ liệu
  - ✅ Chạy training
  - ✅ Lưu kết quả về Drive

### 📝 Cấu hình Training

Sửa trong notebook (cell "Chạy training tự động"):

```python
EPOCHS = 50          # Số epochs
BATCH_SIZE = 32      # Batch size
LEARNING_RATE = 1e-3 # Learning rate
USE_QAT = True       # Quantization-Aware Training
USE_DISTILLATION = True  # Knowledge Distillation
```

### 📁 Kết quả

Sau khi training, kết quả được lưu tại:
- Google Drive: `MyDrive/age_gender_emotion_training/training_YYYYMMDD_HHMMSS/`
- Bao gồm: checkpoints, ONNX model, logs

---

## 🔄 Sync Code với GitHub

### Tự Động Sync

GitHub **KHÔNG** tự động sync theo thời gian thực. Bạn cần commit và push.

#### Cách 1: Script Tự Động

```bash
training_experiments/scripts/auto_sync.bat
```

#### Cách 2: Theo Dõi Real-time

```bash
# Cài đặt watchdog (lần đầu)
pip install watchdog

# Chạy script theo dõi
training_experiments/scripts/watch_sync.bat
```

Script sẽ tự động commit và push khi có thay đổi.

### Setup GitHub cho Colab

1. Sửa trong notebook `train_on_colab_auto.ipynb`:
   ```python
   USE_GITHUB = True
   GITHUB_REPO_URL = "https://github.com/khoiabc2020/age-gender-emotion-detection.git"
   ```

2. Chạy notebook - Tự động pull code mới nhất từ GitHub

---

## ❓ Troubleshooting

### Lỗi: "authentication failed"
- ✅ Đảm bảo dùng **TOKEN** chứ không phải password
- ✅ Kiểm tra token còn hạn không
- ✅ Đảm bảo token có quyền **repo**

### Lỗi: "repository not found"
- ✅ Kiểm tra đã tạo repository trên GitHub chưa
- ✅ Kiểm tra username đúng: `khoiabc2020`
- ✅ Kiểm tra tên repo đúng: `age-gender-emotion-detection`

### Lỗi: "Out of Memory" trên Colab
- ✅ Giảm `BATCH_SIZE` xuống 16 hoặc 8
- ✅ Giảm số epochs để test

### Lỗi: "Module not found" trên Colab
- ✅ Chạy lại cell "Cài đặt dependencies"

---

## 📚 Tài liệu Tham khảo

- Git Documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com/
- Google Colab: https://colab.research.google.com/

---

**Lưu ý**: Token là bí mật, đừng chia sẻ với ai!

