# Hướng dẫn Đóng gói Edge AI App thành .exe

## Tổng quan

Hướng dẫn này sẽ giúp bạn đóng gói Edge AI Application thành file `.exe` có thể cài đặt trên Windows.

## Yêu cầu

### 1. Python Environment
- Python 3.8 - 3.12 (khuyến nghị Python 3.11)
- Đã cài đặt tất cả dependencies từ `requirements.txt`

### 2. Build Tools
- **PyInstaller**: Để tạo file .exe
- **Inno Setup** hoặc **NSIS**: Để tạo installer (tùy chọn)

## Các bước thực hiện

### Bước 1: Chuẩn bị môi trường

```bash
# Di chuyển vào thư mục ai_edge_app
cd ai_edge_app

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt PyInstaller
pip install pyinstaller pyinstaller-hooks-contrib
```

### Bước 2: Kiểm tra Models và Configs

Đảm bảo các file sau tồn tại:
- `models/multitask_model.onnx` - Model chính
- `configs/camera_config.json` - Cấu hình camera
- `configs/ads_rules.json` - Quy tắc quảng cáo

### Bước 3: Build Executable

#### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
python build_exe.py
```

Script này sẽ:
- Tự động kiểm tra và cài đặt dependencies
- Tạo file `.spec` cho PyInstaller
- Build file `.exe` trong thư mục `dist/`

#### Cách 2: Build thủ công

```bash
# Tạo spec file
pyinstaller --name=SmartRetailAI \
    --onefile \
    --windowed \
    --add-data "configs;configs" \
    --add-data "models;models" \
    --hidden-import=PyQt6.QtCore \
    --hidden-import=PyQt6.QtGui \
    --hidden-import=PyQt6.QtWidgets \
    --hidden-import=cv2 \
    --hidden-import=onnxruntime \
    main_gui.py
```

### Bước 4: Kiểm tra Executable

```bash
# Chạy thử file .exe
dist/SmartRetailAI.exe
```

Kiểm tra:
- ✅ Ứng dụng khởi động được
- ✅ Camera hoạt động
- ✅ Models load được
- ✅ GUI hiển thị đúng

### Bước 5: Tạo Installer (Tùy chọn)

#### Option A: Inno Setup (Khuyến nghị cho Windows)

1. Tải Inno Setup: https://jrsoftware.org/isdl.php
2. Chạy script tạo installer:
   ```bash
   python create_installer.py
   ```
3. Mở `installer/SmartRetailAI.iss` trong Inno Setup Compiler
4. Build -> Compile
5. File installer sẽ ở `installer/Output/SmartRetailAI_Setup.exe`

#### Option B: NSIS

1. Tải NSIS: https://nsis.sourceforge.io/Download
2. Chạy script tạo installer:
   ```bash
   python create_installer.py
   ```
3. Right-click `installer/SmartRetailAI.nsi` -> Compile NSIS Script
4. File installer sẽ được tạo trong `installer/`

## Cấu trúc sau khi build

```
ai_edge_app/
├── dist/
│   └── SmartRetailAI.exe          # File .exe chính
├── build/                          # Build artifacts (có thể xóa)
├── installer/
│   ├── SmartRetailAI.iss          # Inno Setup script
│   ├── SmartRetailAI.nsi          # NSIS script
│   └── Output/
│       └── SmartRetailAI_Setup.exe # Installer file
└── ...
```

## Troubleshooting

### Lỗi: "Module not found"

**Giải pháp**: Thêm module vào `hiddenimports` trong file `.spec`:
```python
hiddenimports=[
    'module_name',
    ...
]
```

### Lỗi: "Cannot find models/configs"

**Giải pháp**: Đảm bảo `datas` trong `.spec` bao gồm:
```python
datas=[
    ('configs', 'configs'),
    ('models', 'models'),
]
```

### Lỗi: Camera không hoạt động

**Giải pháp**: 
1. Kiểm tra camera có được kết nối
2. Kiểm tra quyền truy cập camera trong Windows Settings
3. Thử thay đổi `camera_source` trong `camera_config.json`

### File .exe quá lớn

**Giải pháp**: Sử dụng UPX compression (đã bật trong spec):
- UPX sẽ tự động nén các file binary
- Có thể giảm kích thước 30-50%

## Phân phối

### Cách 1: Phân phối file .exe trực tiếp
- Copy `dist/SmartRetailAI.exe` và thư mục `dist/` (nếu có dependencies)
- Người dùng chỉ cần chạy file .exe

### Cách 2: Phân phối qua Installer (Khuyến nghị)
- Sử dụng `SmartRetailAI_Setup.exe`
- Installer sẽ tự động:
  - Tạo shortcut trên Desktop
  - Thêm vào Start Menu
  - Tạo entry trong Add/Remove Programs
  - Cài đặt vào Program Files

## Lưu ý quan trọng

1. **Antivirus**: Một số antivirus có thể flag file .exe. Cần code signing để tránh.
2. **Dependencies**: File .exe đã bao gồm tất cả dependencies, không cần cài Python.
3. **Models**: Đảm bảo models được copy vào thư mục đúng.
4. **Testing**: Luôn test trên máy sạch (không có Python) trước khi phân phối.

## Code Signing (Tùy chọn - cho Production)

Để tránh cảnh báo từ Windows và antivirus:

1. Mua Code Signing Certificate
2. Sign file .exe:
   ```bash
   signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com SmartRetailAI.exe
   ```

## Hỗ trợ

Nếu gặp vấn đề, kiểm tra:
- Logs trong `logs/edge_app.log`
- Console output khi chạy .exe
- Windows Event Viewer
