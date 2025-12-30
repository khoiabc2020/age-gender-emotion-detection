# 🚀 Hướng dẫn Tự Động Upload và Train trên Colab

## Cách 1: Tự động hoàn toàn (Khuyến nghị)

### Bước 1: Chạy script upload
```bash
cd training_experiments
python scripts/upload_to_colab.py
```

Script này sẽ:
- ✅ Tạo file zip từ code
- ✅ Upload lên Google Drive (nếu có credentials)
- ✅ Hoặc tạo file zip để bạn upload thủ công

### Bước 2: Mở Colab và chạy notebook tự động

1. **Mở Google Colab**: https://colab.research.google.com/

2. **Upload notebook**: 
   - File → Upload notebook
   - Chọn file: `training_experiments/notebooks/train_on_colab_auto.ipynb`

3. **Chọn GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator → GPU (T4)
   - Save

4. **Chạy tất cả**:
   - Runtime → Run all
   - Hoặc nhấn Ctrl+F9

Notebook sẽ tự động:
- ✅ Cài đặt dependencies
- ✅ Kiểm tra GPU
- ✅ Mount Google Drive
- ✅ Download code từ Drive
- ✅ Setup dữ liệu
- ✅ Chạy training
- ✅ Lưu kết quả về Drive

## Cách 2: Upload thủ công (Nếu không có Google Drive API)

### Bước 1: Tạo file zip
```bash
cd training_experiments
python scripts/upload_to_colab.py
```

File zip sẽ được tạo tại thư mục gốc: `training_experiments_YYYYMMDD_HHMMSS.zip`

### Bước 2: Upload lên Google Drive
1. Mở Google Drive: https://drive.google.com/
2. Tạo thư mục: `Colab_Training`
3. Upload file zip vào thư mục đó

### Bước 3: Mở Colab
1. Mở: https://colab.research.google.com/
2. Upload notebook: `train_on_colab_auto.ipynb`
3. Chọn GPU runtime
4. Chạy tất cả cells

## Cấu hình Training

Để thay đổi cấu hình training, sửa trong cell "Chạy training tự động":

```python
EPOCHS = 50          # Số epochs
BATCH_SIZE = 32      # Batch size
LEARNING_RATE = 1e-3 # Learning rate
USE_QAT = True       # Quantization-Aware Training
USE_DISTILLATION = True  # Knowledge Distillation
```

## Lưu ý quan trọng

1. **GPU Runtime**: Luôn chọn GPU trước khi chạy
2. **Dữ liệu**: Đảm bảo dữ liệu đã được upload lên Google Drive hoặc có sẵn
3. **Thời gian**: Colab free có thể ngắt sau 90 phút không hoạt động
4. **Kết quả**: Tự động lưu về Google Drive với timestamp

## Troubleshooting

### Lỗi: "Không tìm thấy file zip"
- Kiểm tra file zip đã được upload vào `Colab_Training/` trên Drive
- Hoặc sửa đường dẫn trong cell "Download code từ Google Drive"

### Lỗi: "Out of Memory"
- Giảm `BATCH_SIZE` xuống 16 hoặc 8
- Giảm số epochs

### Lỗi: "Module not found"
- Chạy lại cell "Cài đặt dependencies"

## Kết quả

Sau khi training xong, kết quả sẽ được lưu tại:
- Google Drive: `MyDrive/age_gender_emotion_training/training_YYYYMMDD_HHMMSS/`
- Bao gồm:
  - `checkpoints/best_model.pth` - Model tốt nhất
  - `mobileone_multitask.onnx` - Model ONNX
  - `logs/` - TensorBoard logs

Chúc bạn training thành công! 🎉


