# Hướng dẫn Build và Phân phối Edge AI App

## Tổng quan

Edge AI Application có thể được đóng gói thành file `.exe` độc lập để chạy trên Windows mà không cần cài đặt Python.

## Quick Start

### 1. Build Executable

```bash
cd ai_edge_app
pip install -r requirements.txt
pip install pyinstaller pyinstaller-hooks-contrib
python build_exe.py
```

File `.exe` sẽ được tạo tại: `dist/SmartRetailAI.exe`

### 2. Tạo Installer

```bash
python create_installer.py
```

Sau đó sử dụng Inno Setup hoặc NSIS để compile installer script.

## Chi tiết các bước

### Bước 1: Chuẩn bị

1. **Cài đặt Python 3.8-3.12**
   - Download từ: https://www.python.org/downloads/
   - Đảm bảo chọn "Add Python to PATH"

2. **Cài đặt dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_build.txt
   ```

3. **Kiểm tra Models**
   - Đảm bảo `models/multitask_model.onnx` tồn tại
   - Nếu chưa có, copy từ `training_experiments/models/`

### Bước 2: Build

Chạy script tự động:
```bash
python build_exe.py
```

Hoặc build thủ công:
```bash
pyinstaller --name=SmartRetailAI --onefile --windowed --add-data "configs;configs" --add-data "models;models" main_gui.py
```

### Bước 3: Test

1. Chạy file `.exe`:
   ```bash
   dist\SmartRetailAI.exe
   ```

2. Kiểm tra:
   - ✅ GUI hiển thị
   - ✅ Camera hoạt động
   - ✅ Detection hoạt động
   - ✅ Models load được

### Bước 4: Tạo Installer

1. **Inno Setup** (Khuyến nghị):
   - Download: https://jrsoftware.org/isdl.php
   - Mở `installer/SmartRetailAI.iss`
   - Build -> Compile

2. **NSIS** (Alternative):
   - Download: https://nsis.sourceforge.io/Download
   - Compile `installer/SmartRetailAI.nsi`

## Cấu trúc File

```
SmartRetailAI.exe          # Main executable
├── configs/               # Configuration files
│   ├── camera_config.json
│   └── ads_rules.json
└── models/                # AI Models
    └── multitask_model.onnx
```

## Tính năng

✅ **Standalone**: Không cần Python
✅ **Camera Support**: Tự động detect camera
✅ **Real-time Processing**: Face detection, tracking, classification
✅ **Modern GUI**: PyQt6 interface
✅ **Configurable**: JSON config files
✅ **Logging**: Tự động log vào `logs/`

## Troubleshooting

### Camera không hoạt động
- Kiểm tra camera có được kết nối
- Kiểm tra quyền truy cập trong Windows Settings
- Thử thay đổi `camera_source` trong config

### Model không load
- Đảm bảo `models/multitask_model.onnx` tồn tại
- Kiểm tra đường dẫn trong code

### Lỗi import
- Rebuild với `--hidden-import` cho module bị thiếu

## Phân phối

### Option 1: File .exe đơn giản
- Copy toàn bộ thư mục `dist/`
- Người dùng chạy `SmartRetailAI.exe`

### Option 2: Installer (Khuyến nghị)
- Phân phối `SmartRetailAI_Setup.exe`
- Người dùng cài đặt như phần mềm thông thường
- Tự động tạo shortcuts và uninstaller

## Yêu cầu hệ thống

- **OS**: Windows 10/11 (64-bit)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Camera**: USB webcam hoặc built-in camera
- **Storage**: ~500MB cho app + models

## Lưu ý

1. File .exe đầu tiên có thể bị Windows Defender flag - đây là bình thường
2. Cần code signing certificate để tránh cảnh báo (cho production)
3. Test trên máy sạch trước khi phân phối
