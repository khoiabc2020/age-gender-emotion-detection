# 🚀 Hướng dẫn Training trên Google Colab

Hướng dẫn chi tiết để train model trên Google Colab khi máy tính của bạn không đủ mạnh.

## 📋 Chuẩn bị

### 1. Chuẩn bị code
- Đảm bảo bạn có toàn bộ code trong thư mục `training_experiments`
- Nén thư mục `training_experiments` thành file zip (tùy chọn, để upload dễ hơn)

### 2. Chuẩn bị dữ liệu
Có 3 cách để có dữ liệu trên Colab:

**Option A: Upload từ máy tính** (cho dataset nhỏ < 2GB)
- Nén thư mục `data/processed` thành file zip
- Upload lên Colab

**Option B: Download từ Kaggle** (khuyến nghị)
- Cần có tài khoản Kaggle
- Tạo API token: https://www.kaggle.com/settings -> API -> Create New Token
- Download file `kaggle.json`

**Option C: Upload lên Google Drive trước**
- Upload dữ liệu lên Google Drive
- Mount Drive trong Colab và copy dữ liệu

## 🎯 Các bước thực hiện

### Bước 1: Mở notebook trên Colab

1. Mở file `notebooks/train_on_colab.ipynb` trong Google Colab
   - Cách 1: Upload file `.ipynb` lên Google Drive, mở bằng Colab
   - Cách 2: Copy nội dung notebook vào Colab mới

2. **QUAN TRỌNG**: Chọn GPU runtime
   - Click vào `Runtime` → `Change runtime type`
   - Chọn `Hardware accelerator` → `GPU` (T4 hoặc tốt hơn)
   - Click `Save`

### Bước 2: Chạy các cell theo thứ tự

#### Cell 1: Cài đặt dependencies
- Tự động cài đặt PyTorch, Albumentations, và các thư viện cần thiết
- Chờ đến khi thấy "✅ Đã cài đặt xong các thư viện!"

#### Cell 2: Kiểm tra GPU
- Kiểm tra xem GPU đã được kích hoạt chưa
- Nếu không có GPU, quay lại Bước 1

#### Cell 3: Mount Google Drive
- Cho phép Colab truy cập Google Drive để lưu kết quả
- Click vào link, đăng nhập và copy mã xác thực

#### Cell 4-5: Upload code
- **Cách 1**: Upload thư mục `training_experiments` trực tiếp qua file browser
- **Cách 2**: Upload file zip và giải nén

#### Cell 6-8: Upload/Download dữ liệu
Chọn 1 trong 3 cách:
- **Option A**: Upload file zip chứa dữ liệu đã processed
- **Option B**: Download từ Kaggle (cần upload `kaggle.json`)
- **Option C**: Copy từ Google Drive (nếu đã upload trước đó)

#### Cell 9: Kiểm tra dữ liệu
- Kiểm tra xem dữ liệu đã sẵn sàng chưa
- Đảm bảo có ảnh trong thư mục `train/`, `val/`, `test/`

#### Cell 10-11: Chạy Training
- **Cell 10**: Training với cấu hình mặc định (50 epochs, batch 32)
- **Cell 11**: Training với cấu hình tùy chỉnh (có thể thay đổi epochs, batch size, etc.)

#### Cell 12: Lưu kết quả về Google Drive
- Tự động copy checkpoints, logs, và ONNX model về Google Drive
- Kết quả được lưu với timestamp để dễ quản lý

#### Cell 13-14: Xem kết quả
- Xem metrics training
- Xem biểu đồ trên TensorBoard

## ⚙️ Tùy chỉnh Training

### Thay đổi số epochs
```python
EPOCHS = 50  # Thay đổi số này
```

### Thay đổi batch size
```python
BATCH_SIZE = 32  # Tăng nếu GPU đủ mạnh (64, 128)
```

### Tắt/bật các tính năng
```python
USE_QAT = True          # Quantization-Aware Training
USE_DISTILLATION = True # Knowledge Distillation
```

## 📝 Lưu ý quan trọng

### 1. Thời gian training
- **Colab Free**: ~12 giờ/ngày, có thể bị ngắt kết nối
- **Colab Pro**: ~24 giờ/ngày, GPU tốt hơn
- **Lưu ý**: Luôn lưu checkpoint về Google Drive để không bị mất khi session hết hạn

### 2. Resume training
Nếu training bị gián đoạn, có thể tiếp tục từ checkpoint:

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = Path('/content/project/training_experiments/checkpoints/week2_colab/best_model.pth')
checkpoint = torch.load(checkpoint_path, map_location='cuda')

# Load vào model và optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Tiếp tục training từ start_epoch
```

### 3. Tối ưu cho Colab
- **Batch size**: 32-64 (tùy GPU)
- **num_workers**: 2-4 (Colab có thể không hỗ trợ nhiều workers)
- **Mixed precision**: Tự động bật trong code
- **Giảm epochs**: Nếu muốn test nhanh, giảm xuống 5-10 epochs

### 4. Xử lý lỗi Out of Memory
Nếu gặp lỗi "Out of Memory":
1. Giảm `batch_size` xuống 16 hoặc 8
2. Giảm `image_size` (nếu có)
3. Tắt một số augmentation

### 5. Download kết quả
Sau khi training xong:
1. Vào Google Drive: `MyDrive/age_gender_emotion_training/`
2. Tìm thư mục `training_YYYYMMDD_HHMMSS/`
3. Download các file:
   - `checkpoints/best_model.pth` - Model tốt nhất
   - `mobileone_multitask.onnx` - Model ONNX để deploy
   - `logs/` - TensorBoard logs

## 🔧 Troubleshooting

### Lỗi: "ModuleNotFoundError"
- Chạy lại cell cài đặt dependencies
- Đảm bảo đã chạy tất cả các cell theo thứ tự

### Lỗi: "CUDA out of memory"
- Giảm `batch_size` xuống 16 hoặc 8
- Giảm số epochs để test

### Lỗi: "File not found"
- Kiểm tra đường dẫn file
- Đảm bảo đã upload code và dữ liệu đúng vị trí

### Training quá chậm
- Kiểm tra GPU đã được kích hoạt chưa
- Tăng `num_workers` (nhưng không quá 4)
- Giảm số epochs để test nhanh

### Session bị ngắt
- Colab free có thể ngắt kết nối sau 90 phút không hoạt động
- Luôn lưu checkpoint về Google Drive
- Sử dụng Colab Pro để có thời gian dài hơn

## 📊 Monitor Training

### TensorBoard
Chạy cell TensorBoard để xem:
- Loss curves
- Accuracy metrics
- Learning rate schedule

### Print logs
Training script sẽ in ra:
- Loss mỗi epoch
- Validation metrics
- Best model được lưu khi nào

## 🎉 Hoàn tất

Sau khi training xong:
1. ✅ Model được lưu tại `checkpoints/week2_colab/best_model.pth`
2. ✅ ONNX model tại `checkpoints/week2_colab/mobileone_multitask.onnx`
3. ✅ Tất cả đã được backup lên Google Drive
4. ✅ Có thể download về máy để sử dụng

Chúc bạn training thành công! 🚀


