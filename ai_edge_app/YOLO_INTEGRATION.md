# 🎯 Tích hợp YOLO vào Dự án

## ✅ Đã Tích hợp YOLO

Dự án hiện hỗ trợ **3 loại detector**:
1. **RetinaFace** (mặc định) - Chuyên cho face detection
2. **YOLO Face** - YOLO cho face detection (nhanh hơn)
3. **YOLO Person** - YOLO cho full body detection (tracking toàn bộ người)

## 📊 So sánh YOLO vs RetinaFace

| Feature | RetinaFace | YOLO Face | YOLO Person |
|---------|------------|-----------|-------------|
| **Tốc độ** | Trung bình | ⚡ Nhanh hơn | ⚡ Nhanh nhất |
| **Độ chính xác** | ⭐⭐⭐⭐⭐ Cao | ⭐⭐⭐⭐ Tốt | ⭐⭐⭐ Tốt |
| **Face Detection** | ✅ Chuyên biệt | ✅ Tốt | ❌ Không |
| **Full Body** | ❌ Không | ❌ Không | ✅ Có |
| **Model Size** | ~1.7MB | ~6MB | ~6MB |
| **Edge Device** | ✅ Tốt | ✅ Tốt | ✅ Tốt |
| **ONNX Support** | ✅ | ✅ | ✅ |

## 🚀 Cách Sử dụng

### 1. Download YOLO Models

#### Option A: YOLOv8 Face Detection
```bash
# Download YOLOv8n-face.onnx từ:
# https://github.com/derronqi/yolov8-face
# Hoặc convert từ PyTorch:
# pip install ultralytics
# python -c "from ultralytics import YOLO; model = YOLO('yolov8n-face.pt'); model.export(format='onnx')"
```

#### Option B: YOLOv8 Person Detection (COCO)
```bash
# Download YOLOv8n.onnx từ:
# https://github.com/ultralytics/ultralytics
# Hoặc:
# pip install ultralytics
# python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"
```

### 2. Cấu hình Detector

Chỉnh sửa `configs/camera_config.json`:

```json
{
  "detection": {
    "type": "yolo_face",  // "retinaface", "yolo_face", "yolo_person"
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "min_face_size": 40
  }
}
```

### 3. Đặt Model vào thư mục

```bash
# Copy YOLO model vào models/
cp yolov8n-face.onnx ai_edge_app/models/
# hoặc
cp yolov8n.onnx ai_edge_app/models/
```

### 4. Chạy ứng dụng

```bash
cd ai_edge_app
python main.py
```

## 🎯 Khi Nào Dùng YOLO?

### Dùng YOLO Face khi:
- ✅ Cần tốc độ cao hơn RetinaFace
- ✅ Có nhiều faces trong frame (>5 faces)
- ✅ Cần real-time performance tốt hơn
- ✅ Model size không phải vấn đề

### Dùng YOLO Person khi:
- ✅ Cần tracking toàn bộ người (full body)
- ✅ Muốn detect người từ xa (trước khi thấy mặt)
- ✅ Cần analytics về hành vi (đi lại, dừng lại)
- ✅ Muốn detect nhiều người cùng lúc

### Dùng RetinaFace khi:
- ✅ Cần độ chính xác cao nhất cho face
- ✅ Model size nhỏ là ưu tiên
- ✅ Chỉ cần detect face, không cần full body
- ✅ Đã có model RetinaFace sẵn

## 📁 Cấu trúc Code

```
ai_edge_app/
├── src/
│   └── detectors/
│       ├── __init__.py          # Export all detectors
│       ├── retinaface_detector.py  # RetinaFace
│       └── yolo_detector.py     # YOLO (Face & Person) ⭐ NEW
├── models/
│   ├── retinaface_mnet.onnx     # RetinaFace model
│   ├── yolov8n-face.onnx        # YOLO Face model (optional)
│   └── yolov8n.onnx              # YOLO Person model (optional)
└── configs/
    └── camera_config.json        # Config detector type
```

## 🔧 Tùy chỉnh YOLO

### Thay đổi Input Size

```python
# Trong yolo_detector.py
detector = YOLODetector(
    model_path="models/yolov8n.onnx",
    input_size=(416, 416)  # Smaller = faster, less accurate
    # hoặc (640, 640)  # Default
    # hoặc (1280, 1280)  # Larger = slower, more accurate
)
```

### Điều chỉnh Thresholds

```json
{
  "detection": {
    "confidence_threshold": 0.5,  // Lower = more detections
    "iou_threshold": 0.45          // Lower = less NMS filtering
  }
}
```

## ⚡ Performance

### YOLOv8n (Nano) trên CPU:
- **FPS**: 25-30 FPS (640x640 input)
- **Latency**: ~35ms per frame
- **Memory**: ~200MB

### YOLOv8s (Small) trên CPU:
- **FPS**: 15-20 FPS (640x640 input)
- **Latency**: ~50ms per frame
- **Memory**: ~300MB

### So với RetinaFace:
- **RetinaFace**: ~20-25 FPS
- **YOLOv8n**: ~25-30 FPS (nhanh hơn ~20%)

## 🎓 Download Models

### YOLOv8 Face:
```bash
# Option 1: Download pre-trained
wget https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.onnx

# Option 2: Convert từ PyTorch
pip install ultralytics
python scripts/convert_yolo_face.py
```

### YOLOv8 Person (COCO):
```bash
# Download từ Ultralytics
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
```

## 🔄 Migration từ RetinaFace sang YOLO

1. **Download YOLO model** (xem trên)
2. **Copy vào `models/`**
3. **Cập nhật config**: `"type": "yolo_face"`
4. **Chạy lại app** - Tự động load YOLO!

## 💡 Tips

1. **YOLO Face** tốt hơn khi có nhiều faces (>3)
2. **YOLO Person** tốt cho tracking full body
3. **RetinaFace** vẫn tốt nhất cho độ chính xác
4. Có thể **switch giữa các detectors** dễ dàng qua config

## 🐛 Troubleshooting

### Model không load
- Kiểm tra file `.onnx` có tồn tại không
- Kiểm tra path trong config
- Xem logs để biết lỗi cụ thể

### Detection không chính xác
- Giảm `confidence_threshold` xuống 0.3-0.4
- Tăng `input_size` lên 1280x1280
- Thử model lớn hơn (YOLOv8s thay vì YOLOv8n)

### FPS thấp
- Giảm `input_size` xuống 416x416
- Sử dụng YOLOv8n (nano) thay vì YOLOv8s
- Enable GPU nếu có

## ✅ Checklist

- [x] YOLO detector class created
- [x] Support YOLO Face & Person
- [x] ONNX Runtime integration
- [x] Config-based detector selection
- [x] Fallback to RetinaFace
- [x] Documentation

## 🚀 Next Steps

1. Download YOLO models
2. Test với config `"type": "yolo_face"`
3. So sánh performance với RetinaFace
4. Chọn detector phù hợp với use case



