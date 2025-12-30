# Edge AI Application - Smart Retail Analytics

Ứng dụng Edge AI chạy trên camera/laptop để nhận diện khách hàng realtime và đề xuất quảng cáo.

## 🎯 Giai đoạn 2: Edge Client Application (Tuần 5-7)

### ✅ Tuần 5: Face Detection & Tracking Pipeline

- **RetinaFace Detector**: Face detection với ONNX Runtime
  - Hỗ trợ ONNX model
  - Fallback: OpenCV DNN hoặc Haar Cascade
  - Optimized cho edge devices

- **DeepSORT Tracker**: Multi-face tracking
  - IoU matching
  - Track prediction và smoothing
  - Xử lý track aging và confirmation

### ✅ Tuần 6: Model & Ads Engine

- **MultiTaskClassifier**: Age, Gender, Emotion classification
  - Sử dụng ONNX model từ Giai đoạn 1
  - Preprocessing chuẩn ImageNet
  - Post-processing với softmax

- **Ads Selector**: Hybrid advertisement selection
  - Filtering: Lọc theo age, gender, emotion
  - Scoring: Context score + Emotion score
  - Exploration: 10% random cho A/B testing

- **UI Display**: OpenCV-based display
  - Real-time visualization
  - Track info overlay
  - FPS monitoring

### ✅ Tuần 7: Performance Optimization

- **Threading Support**: Frame buffering
- **FPS Monitoring**: Real-time FPS tracking
- **Caching**: Cache attributes để giảm computation

## 🚀 Quick Start

### 1. Setup

```bash
cd ai_edge_app

# Install dependencies
pip install -r requirements.txt

# Copy models từ training_experiments
cp ../training_experiments/models/multitask_efficientnet.onnx models/
# (RetinaFace model sẽ được download hoặc sử dụng fallback)
```

### 2. Configure

Chỉnh sửa `configs/camera_config.json`:
```json
{
  "camera": {
    "source": 0,  // 0 = webcam, hoặc đường dẫn video file
    "width": 640,
    "height": 480
  }
}
```

### 3. Run

```bash
python main.py
```

Nhấn `q` để thoát.

## 📁 Cấu trúc

```
ai_edge_app/
├── main.py                    # Entry point
├── configs/
│   ├── camera_config.json     # Camera & detection config
│   └── ads_rules.json         # Advertisement rules
├── models/                    # ONNX models
│   ├── retinaface_mnet.onnx   # (Optional)
│   └── multitask_efficientnet.onnx
├── src/
│   ├── detectors/            # Face detection
│   │   └── retinaface_detector.py
│   ├── trackers/             # Face tracking
│   │   └── deepsort_tracker.py
│   ├── classifiers/          # Attribute classification
│   │   └── multitask_classifier.py
│   ├── ads_engine/           # Ad selection logic
│   │   └── ads_selector.py
│   └── utils/                # Utilities
│       ├── logger.py
│       └── mqtt_client.py
└── logs/                     # Application logs
```

## ⚙️ Configuration

### Camera Config (`configs/camera_config.json`)

```json
{
  "camera": {
    "source": 0,              // Camera index hoặc video file path
    "width": 640,
    "height": 480,
    "fps": 30
  },
  "detection": {
    "confidence_threshold": 0.7,
    "min_face_size": 40
  },
  "tracking": {
    "max_age": 30,            // Frames to keep track without detection
    "min_hits": 3,            // Min detections to confirm track
    "iou_threshold": 0.3      // IoU threshold for matching
  },
  "mqtt": {
    "broker": "localhost",
    "port": 1883,
    "topic": "retail/analytics",
    "device_key": "edge_device_001"
  }
}
```

### Ads Rules (`configs/ads_rules.json`)

Xem file mẫu để biết cấu trúc rules.

## 📊 Performance

- **Target FPS**: > 15 FPS trên laptop Core i5
- **Latency**: < 200ms từ detection đến ad selection
- **Memory**: ~500MB RAM

## 🔧 Troubleshooting

### Camera không mở được

- Kiểm tra camera index (thử 0, 1, 2...)
- Kiểm tra quyền truy cập camera
- Thử với video file thay vì camera

### Model không load

- Kiểm tra file model có tồn tại trong `models/`
- Copy model từ `training_experiments/models/`
- RetinaFace sẽ tự động fallback nếu không có model

### FPS thấp

- Giảm resolution trong config
- Tăng interval giữa các lần classification (hiện tại 2 giây)
- Sử dụng GPU nếu có (cần cấu hình ONNX Runtime)

### MQTT connection failed

- Ứng dụng vẫn chạy được, chỉ không gửi analytics
- Kiểm tra MQTT broker có chạy không
- Có thể bỏ qua nếu không cần real-time analytics

## 🔄 Next Steps

Sau khi hoàn thành Giai đoạn 2:
1. Test với camera thật
2. Tối ưu FPS nếu cần
3. Bắt đầu Giai đoạn 3: Backend API & Database

