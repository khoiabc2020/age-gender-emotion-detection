# 🚀 Training Tự Động trên Google Colab

## ⚡ Cách nhanh nhất (3 bước)

### Bước 1: Chạy script upload
```bash
# Windows
CHAY_TU_DONG_COLAB.bat

# Hoặc Python
python scripts/upload_to_colab.py
```

### Bước 2: Mở Colab
1. Truy cập: https://colab.research.google.com/
2. Upload file: `notebooks/train_on_colab_auto.ipynb`
3. Chọn GPU: Runtime → Change runtime type → GPU

### Bước 3: Chạy tự động
- Runtime → Run all (hoặc Ctrl+F9)
- Đợi training hoàn tất!

## 📋 Chi tiết

### Script upload (`scripts/upload_to_colab.py`)
- Tạo file zip từ code
- Upload lên Google Drive (nếu có credentials)
- Hoặc tạo file zip để upload thủ công

### Notebook tự động (`notebooks/train_on_colab_auto.ipynb`)
Tự động thực hiện:
1. ✅ Cài đặt dependencies
2. ✅ Kiểm tra GPU
3. ✅ Mount Google Drive
4. ✅ Download code từ Drive
5. ✅ Setup dữ liệu
6. ✅ Chạy training
7. ✅ Lưu kết quả về Drive

## ⚙️ Tùy chỉnh

Sửa trong notebook (cell "Chạy training tự động"):
```python
EPOCHS = 50          # Số epochs
BATCH_SIZE = 32      # Batch size
LEARNING_RATE = 1e-3 # Learning rate
```

## 📁 Kết quả

Sau khi training, kết quả được lưu tại:
- Google Drive: `MyDrive/age_gender_emotion_training/training_YYYYMMDD_HHMMSS/`
- Bao gồm: checkpoints, ONNX model, logs

## 🔧 Troubleshooting

- **Không tìm thấy file zip**: Upload file zip vào `Colab_Training/` trên Drive
- **Out of Memory**: Giảm `BATCH_SIZE` xuống 16 hoặc 8
- **Module not found**: Chạy lại cell "Cài đặt dependencies"

Xem thêm: `notebooks/HUONG_DAN_TU_DONG.md`


