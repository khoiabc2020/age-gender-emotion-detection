# 📅 KẾ HOẠCH THỰC THI CHI TIẾT

**Dự án:** Smart Retail Analytics & Targeted Ads  
**Tổng thời gian:** 15-16 tuần  
**Mục tiêu:** Hoàn thành đồ án tốt nghiệp mức độ Xuất sắc (A+)

---

## 🛑 GIAI ĐOẠN 1: XÂY DỰNG LÕI AI (AI CORE DEVELOPMENT)

**Thời gian:** Tuần 1 - Tuần 4  
**Mục tiêu:** Có được file model .pth (hoặc .onnx) nhận diện chính xác Tuổi, Giới tính, Cảm xúc.

### ✅ Tuần 1: Chuẩn bị Dữ liệu (Data Preparation)

**Đã hoàn thành:**
- ✅ Tải 3 datasets: UTKFace, FER2013, All Age Face
- ✅ Script preprocessing (`src/data/preprocess.py`)
- ✅ DataLoader với Albumentations (`src/data/dataset.py`)

**Cần làm:**
```bash
cd training_experiments
python src/data/preprocess.py  # Preprocess datasets
python src/data/dataset.py      # Test DataLoader
```

### 📌 Tuần 2: Xây dựng Kiến trúc Model (Model Architecture)

**Đã hoàn thành:**
- ✅ Multi-task Model với EfficientNet-B0 (`src/models/network.py`)
- ✅ Loss function kết hợp (`MultiTaskLoss`)

**Cần làm:**
```bash
python src/models/network.py  # Test model architecture
```

**Kiến trúc:**
- Backbone: EfficientNet-B0 (Pre-trained ImageNet)
- Head 1: Gender Classification (2 classes)
- Head 2: Age Regression
- Head 3: Emotion Classification (7 classes)

### 📌 Tuần 3: Huấn luyện & Đánh giá (Training & Eval)

**Đã hoàn thành:**
- ✅ Training script với TensorBoard (`train.py`)
- ✅ Validation và metrics tracking

**Cần làm:**
```bash
python train.py --data_dir data/processed/utkface --batch_size 32 --epochs 50 --lr 1e-3
tensorboard --logdir checkpoints/logs
```

**Mục tiêu:**
- Gender Accuracy: > 92%
- Age MAE: < 5.0 years
- Emotion Accuracy: > 75%

### 📌 Tuần 4: Chuyển đổi & Tối ưu (Optimization)

**Đã hoàn thành:**
- ✅ Script convert sang ONNX (`scripts/convert_to_onnx.py`)
- ✅ Script test inference (`scripts/predict_test.py`)

**Cần làm:**
```bash
python scripts/convert_to_onnx.py --model_path checkpoints/best_model.pth --output_path models/multitask_efficientnet_int8.onnx
python scripts/predict_test.py --model_path models/multitask_efficientnet_int8.onnx --image_path path/to/image.jpg
```

---

## 🛑 GIAI ĐOẠN 2: ỨNG DỤNG EDGE CLIENT (CAMERA APP)

**Thời gian:** Tuần 5 - Tuần 7  
**Mục tiêu:** Chạy được ứng dụng trên Laptop/PC, nhận diện realtime qua Webcam.

### 📌 Tuần 5: Face Detection & Tracking pipeline
- Tích hợp RetinaFace
- Tích hợp DeepSORT
- Gắn ID cho từng khuôn mặt

### 📌 Tuần 6: Ghép nối Model & Logic Quảng cáo
- Xử lý luồng Video
- Xây dựng Ads Engine
- Hiển thị UI với cv2.imshow

### 📌 Tuần 7: Tối ưu hiệu năng
- Đo FPS (mục tiêu > 15 FPS)
- Xử lý đa luồng (Threading)

---

## 🛑 GIAI ĐOẠN 3: BACKEND API & DATABASE

**Thời gian:** Tuần 8 - Tuần 10  
**Mục tiêu:** Lưu trữ lịch sử khách hàng và phục vụ dữ liệu cho Dashboard.

### 📌 Tuần 8: Thiết kế Database & Setup Server
- Setup PostgreSQL
- FastAPI project structure
- SQLAlchemy ORM

### 📌 Tuần 9: Viết API Endpoints
- POST /api/v1/logs (ghi log)
- GET /api/v1/stats (thống kê)

### 📌 Tuần 10: Kết nối Edge với Backend
- Gửi data từ Edge Client lên Server
- Async data transmission

---

## 🛑 GIAI ĐOẠN 4: FRONTEND DASHBOARD

**Thời gian:** Tuần 11 - Tuần 13  
**Mục tiêu:** Giao diện quản trị chuyên nghiệp để báo cáo.

### 📌 Tuần 11: Setup ReactJS & UI Base
- Khởi tạo React project
- Layout với Ant Design

### 📌 Tuần 12: Visualize Dữ liệu
- Tích hợp Recharts
- Kết nối API với Axios

### 📌 Tuần 13: Trang Quản lý & Realtime
- Quản lý quảng cáo
- WebSocket cho realtime updates

---

## 🛑 GIAI ĐOẠN 5: ĐÓNG GÓI & VIẾT BÁO CÁO

**Thời gian:** Tuần 14 - Tuần 16  
**Mục tiêu:** Hoàn thiện sản phẩm để bảo vệ.

### 📌 Tuần 14: Docker hóa
- Dockerfile cho các services
- docker-compose.yml

### 📌 Tuần 15: Viết Báo cáo & Slide
- Thuyết minh đồ án
- Slide PowerPoint

### 📌 Tuần 16: Rehearsal
- Quay video Demo
- Bug hunting

---

## 🚀 GIAI ĐOẠN 6 (OPTIONAL): NÂNG CẤP "NEXT-LEVEL"

**Thời gian:** Làm thêm nếu còn dư thời gian

### 🌟 1. Tích hợp Generative AI (LLM Analyst)
- Chat với dữ liệu
- AI viết báo cáo tự động

### 🌟 2. Nhận diện Khách quen (Face Recognition)
- ArcFace embedding
- Vector Database (Qdrant/Milvus)

### 🌟 3. Kiến trúc Event-Driven
- Message Queue (RabbitMQ/Kafka)
- Xử lý hàng nghìn camera

---

## 💡 MẸO QUAN TRỌNG

1. **Version Control**: Dùng Git ngay từ đầu, commit hàng ngày
2. **Đừng cầu toàn**: Có hệ thống chạy trọn vẹn quan trọng hơn module hoàn hảo
3. **Edge Computing**: Nhấn mạnh ONNX và tối ưu hóa cho laptop bình thường
4. **Mock Data**: Dùng dữ liệu giả khi làm Dashboard nếu chưa kết nối Backend

---

## 📊 Tiến độ hiện tại

- ✅ **Giai đoạn 1 - Tuần 1**: Hoàn thành (Code đã sẵn sàng)
- ⏳ **Giai đoạn 1 - Tuần 2-4**: Cần chạy training và convert model
- ⏳ **Giai đoạn 2-5**: Chưa bắt đầu

**Bước tiếp theo:** Chạy preprocessing và bắt đầu training!

