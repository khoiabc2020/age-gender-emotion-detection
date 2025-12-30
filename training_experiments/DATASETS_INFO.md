# Datasets Information

Thông tin về các datasets đã tải xuống cho dự án Smart Retail Analytics.

## 📊 Danh sách Datasets

### 1. All Age Face Dataset
- **Kaggle ID**: `eshachakraborty00/all-age-face-dataset`
- **Mô tả**: Dataset chứa hình ảnh khuôn mặt với nhãn độ tuổi
- **Kích thước**: ~348 MB
- **Vị trí**: `C:\Users\LE HUY KHOI\.cache\kagglehub\datasets\eshachakraborty00\all-age-face-dataset\versions\1`
- **Cấu trúc**: 
  - `All-Age-Faces Dataset/` - Thư mục chứa dữ liệu

### 2. UTKFace Dataset
- **Kaggle ID**: `jangedoo/utkface-new`
- **Mô tả**: Dataset lớn với nhãn độ tuổi, giới tính và dân tộc
- **Kích thước**: ~331 MB
- **Vị trí**: `C:\Users\LE HUY KHOI\.cache\kagglehub\datasets\jangedoo\utkface-new\versions\1`
- **Cấu trúc**:
  - `crop_part1/` - 9,780 ảnh
  - `UTKFace/` - 23,708 ảnh
  - `utkface_aligned_cropped/` - Ảnh đã được align và crop

### 3. FER2013 Dataset
- **Kaggle ID**: `msambare/fer2013`
- **Mô tả**: Dataset nhận diện cảm xúc với 7 lớp cảm xúc
- **Kích thước**: ~60.3 MB
- **Vị trí**: `C:\Users\LE HUY KHOI\.cache\kagglehub\datasets\msambare\fer2013\versions\1`
- **Cấu trúc**:
  - `train/` - 7 thư mục (mỗi thư mục là một emotion class)
  - `test/` - 7 thư mục (mỗi thư mục là một emotion class)
- **Emotion Classes**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

## 🔧 Sử dụng Datasets

### Tải lại datasets

```bash
cd training_experiments/scripts
python download_all_age_dataset.py
```

### Kiểm tra datasets

```bash
cd training_experiments/scripts
python check_datasets.py
```

### Truy cập datasets trong code

```python
import kagglehub

# All Age Face Dataset
all_age_path = kagglehub.dataset_download("eshachakraborty00/all-age-face-dataset")

# UTKFace Dataset
utkface_path = kagglehub.dataset_download("jangedoo/utkface-new")

# FER2013 Dataset
fer2013_path = kagglehub.dataset_download("msambare/fer2013")
```

## 📝 Ghi chú

- Tất cả datasets được lưu trong Kaggle cache directory
- Datasets sẽ tự động được cache, không cần tải lại nếu đã có
- Để xóa cache và tải lại: Xóa thư mục `.cache/kagglehub/`

## 🎯 Mục đích sử dụng

1. **All Age Face Dataset**: Training model nhận diện độ tuổi
2. **UTKFace Dataset**: Training model nhận diện độ tuổi và giới tính (multi-task learning)
3. **FER2013 Dataset**: Training model nhận diện cảm xúc

## 📈 Thống kê tổng hợp

- **Tổng số datasets**: 3
- **Tổng kích thước**: ~740 MB
- **Tổng số ảnh**: ~33,000+ ảnh (ước tính)

