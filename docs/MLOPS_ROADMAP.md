# 📅 ROADMAP: SMART RETAIL AI (HYBRID MLOPS & EDGE ULTRA)

**Kiến trúc**: Hybrid (Xử lý tại Edge + Quản trị tại Cloud)  
**Công nghệ**: PyQt6, FastViT, Kubernetes, Kubeflow, Kafka, Spark  
**Version**: 4.0.0 Hybrid MLOps Edition

---

## 🎯 TỔNG QUAN

Hệ thống Smart Retail AI được nâng cấp lên kiến trúc Hybrid MLOps, kết hợp:
- **Edge Computing**: Xử lý real-time tại thiết bị
- **Cloud Infrastructure**: Quản lý, phân tích và ML pipeline tự động
- **MLOps**: Tự động hóa training, deployment và monitoring

---

## 🛑 GIAI ĐOẠN 1: CLOUD INFRASTRUCTURE (KUBERNETES)

**Mục tiêu**: Xây dựng "trụ sở chỉ huy" để quản lý dữ liệu và model cho hàng loạt thiết bị Edge.

### 📌 Tuần 1: Setup Kubernetes Cluster & Storage

#### Kubernetes Local
- [x] K8s manifests (`k8s/namespace.yaml`)
- [ ] Cài đặt Kind hoặc Minikube
- [ ] Deploy namespace `smart-retail`

#### Data Lake (MinIO)
- [x] MinIO deployment (`k8s/minio/deployment.yaml`)
- [x] Buckets initialization (`k8s/minio/buckets-init.yaml`)
- [ ] Buckets: `retail-data`, `models`, `training-data`
- [ ] Access: `http://localhost:30090` (API), `http://localhost:30091` (Console)

#### Analytics DB (Elasticsearch)
- [x] Elasticsearch StatefulSet (`k8s/elasticsearch/deployment.yaml`)
- [ ] Access: `http://localhost:30200`
- [ ] Index: `retail-analytics`

### 📌 Tuần 2: Event Streaming Backbone (Kafka)

#### Kafka Cluster
- [x] Kafka cluster config (`k8s/kafka/kafka-cluster.yaml`)
- [ ] Deploy Strimzi Operator
- [ ] Topics: `edge-telemetry`, `edge-images` (`k8s/kafka/topics.yaml`)

---

## 🛑 GIAI ĐOẠN 2: EDGE AI SUPER-APP (CLIENT)

**Mục tiêu**: Ứng dụng tại cửa hàng xử lý cực nhanh, giao diện đẹp, thông minh.

### 📌 Tuần 3: Core AI & Optimization

- [ ] Model: Train FastViT hoặc MobileOne (SOTA Lightweight)
- [ ] Convert sang ONNX/TensorRT
- [x] Modules: MiniFASNet (Anti-spoofing), ByteTrack (Tracking)
- [ ] Logic: Xử lý toàn bộ logic nhận diện và hiển thị quảng cáo Offline tại thiết bị (đảm bảo độ trễ < 200ms)

### 📌 Tuần 4: Modern UI (PyQt6)

- [x] Interface: QFluentWidgets thiết kế giao diện Windows 11
- [ ] Features: Dashboard HUD, Biểu đồ Real-time, Panel Quảng cáo động

### 📌 Tuần 5: Edge-to-Cloud Connector

- [x] Kafka Producer (`ai_edge_app/src/services/kafka_producer.py`)
- [x] Logic gửi data:
  - [x] Gửi Metadata (JSON): Tuổi, Giới tính, Cảm xúc → Kafka topic `edge-telemetry`
  - [ ] Gửi Ảnh (khi độ tin cậy thấp): Upload ảnh khó nhận diện lên MinIO để server học lại

---

## 🛑 GIAI ĐOẠN 3: DATA PROCESSING & ANALYTICS

**Mục tiêu**: Xử lý luồng dữ liệu khổng lồ từ Edge gửi về theo thời gian thực.

### 📌 Tuần 6: Spark Streaming Jobs

- [x] Spark Streaming job (`spark/jobs/streaming_analytics.py`)
- [ ] Deploy Spark Operator trên K8s (`k8s/spark/spark-streaming-job.yaml`)
- [ ] Job 1 (Real-time Analytics):
  - [ ] Đọc từ Kafka → Tính toán (Ví dụ: "Đang có bao nhiêu khách vui vẻ?") → Ghi vào Elasticsearch
- [ ] Job 2 (Data Archiving):
  - [ ] Đọc từ Kafka → Ghi xuống MinIO (định dạng Parquet) để làm dữ liệu train cho Kubeflow

### 📌 Tuần 7: Central Dashboard

- [ ] Grafana / Kibana: Kết nối với Elasticsearch để vẽ biểu đồ tổng hợp toàn hệ thống (All Stores Performance)

---

## 🛑 GIAI ĐOẠN 4: AUTOMATED MLOPS (KUBEFLOW)

**Mục tiêu**: Hệ thống tự động thông minh hơn theo thời gian (Continuous Learning).

### 📌 Tuần 8: Kubeflow Ecosystem

- [ ] Cài đặt Kubeflow (Pipelines, Katib, Notebooks)

### 📌 Tuần 9: Auto-Retraining Pipeline

- [x] Pipeline definition (`kubeflow/pipelines/retraining_pipeline.py`)
- [ ] Pipeline steps:
  - [ ] Data Prep: Lấy dữ liệu mới từ MinIO (do Spark ghi vào)
  - [ ] Training: Dùng Kubeflow Training Operator để fine-tune model FastViT với dữ liệu mới
  - [ ] Evaluation: So sánh độ chính xác với model cũ
  - [ ] Register: Lưu model mới vào Model Registry (MinIO) nếu tốt hơn

### 📌 Tuần 10: Hyperparameter Tuning (Katib)

- [ ] Dùng Katib để tự động tìm tham số (Learning rate, Batch size) tối ưu nhất cho đợt train mới

---

## 🛑 GIAI ĐOẠN 5: MODEL SERVING & SYNC

**Mục tiêu**: Cập nhật trí thông minh mới nhất xuống thiết bị Edge.

### 📌 Tuần 11: KServe Deployment

- [x] KServe config (`k8s/kserve/model-serving.yaml`)
- [ ] Deploy model mới nhất lên KServe để tạo API (dùng cho các tác vụ cần server xử lý hoặc làm benchmark)

### 📌 Tuần 12: Model OTA Update (Over-the-Air)

- [x] OTA Service (`ai_edge_app/src/services/model_ota.py`)
- [ ] Tính năng cho App Edge:
  - [x] Khi khởi động, tự kiểm tra trên Server (MinIO/KServe) xem có model version mới không
  - [x] Nếu có → Tự động tải về và hot-swap (thay thế nóng) model cũ

---

## 🛑 GIAI ĐOẠN 6: ADVANCED FEATURES (GENAI & PACKAGING)

**Mục tiêu**: Tính năng "Sát thủ" và Đóng gói.

### 📌 Tuần 13: GenAI Integration (Tại Edge)

- [ ] Tích hợp Phi-3 Mini (Local LLM) để sinh nội dung quảng cáo cá nhân hóa offline
- [x] Điều khiển không chạm (Hand Gesture) - `ai_edge_app/src/gesture/gesture_recognizer.py`

### 📌 Tuần 14: Packaging & Defense

- [ ] Đóng gói App Edge thành .exe (PyInstaller)
- [ ] Quay video demo quy trình khép kín:
  - [ ] Khách hàng tương tác tại Edge
  - [ ] Dữ liệu bay về Kafka → Spark → Dashboard nhảy số
  - [ ] Kubeflow tự động chạy pipeline train lại model
  - [ ] Edge tải model mới

---

## 💡 KIẾN TRÚC CÔNG NGHỆ (TECH STACK)

| Layer | Công nghệ | Vai trò |
|-------|----------|---------|
| **Edge Device** | PyQt6 + ONNX Runtime | Chạy App, AI Inference, Hiển thị Ads |
| **Messaging** | Kafka | Đường ống truyền dữ liệu siêu tốc Edge <-> Cloud |
| **Processing** | Spark Streaming | Xử lý dữ liệu Streaming, đẩy vào kho lưu trữ |
| **Data Lake** | MinIO | Lưu trữ ảnh raw, dữ liệu train (Parquet), Model |
| **MLOps** | Kubeflow (KFP, Katib) | Tự động hóa quy trình Train & Tối ưu tham số |
| **Analytics** | Elasticsearch | Lưu trữ chỉ số để vẽ biểu đồ quản trị trung tâm |
| **Model Serving** | KServe | Phục vụ model qua API |
| **Orchestration** | Kubernetes | Quản lý và điều phối toàn bộ hệ thống |

---

## 📁 CẤU TRÚC FILE MỚI

```
Smart-Retail-AI/
├── k8s/                      # Kubernetes manifests
│   ├── namespace.yaml
│   ├── minio/
│   ├── kafka/
│   ├── elasticsearch/
│   ├── spark/
│   ├── kubeflow/
│   └── kserve/
├── spark/                    # Spark jobs
│   └── jobs/
│       └── streaming_analytics.py
├── kubeflow/                 # Kubeflow pipelines
│   └── pipelines/
│       └── retraining_pipeline.py
├── ai_edge_app/
│   └── src/services/
│       ├── kafka_producer.py  # Kafka integration
│       └── model_ota.py       # OTA updates
└── ...
```

---

## 🚀 DEPLOYMENT

### Local Development

```bash
# 1. Setup Kubernetes (Kind/Minikube)
kind create cluster --name retail-cluster

# 2. Deploy infrastructure
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/minio/
kubectl apply -f k8s/kafka/
kubectl apply -f k8s/elasticsearch/

# 3. Deploy Spark job
kubectl apply -f k8s/spark/

# 4. Deploy Kubeflow pipeline
kubectl apply -f k8s/kubeflow/

# 5. Deploy KServe
kubectl apply -f k8s/kserve/
```

### Production

- [ ] Setup production Kubernetes cluster (EKS, GKE, AKS)
- [ ] Configure persistent storage
- [ ] Setup monitoring (Prometheus, Grafana)
- [ ] Configure auto-scaling
- [ ] Setup backup & disaster recovery

---

## 📊 MONITORING & OBSERVABILITY

- [ ] Prometheus metrics collection
- [ ] Grafana dashboards
- [ ] ELK stack for logs
- [ ] Distributed tracing (Jaeger)
- [ ] Model performance monitoring

---

## 🔐 SECURITY

- [ ] TLS/SSL certificates
- [ ] Network policies
- [ ] RBAC configuration
- [ ] Secrets management (Vault)
- [ ] Image scanning

---

## ✅ STATUS

**Current Phase**: Giai đoạn 1-2 (Infrastructure & Edge App)  
**Completion**: ~40%  
**Next Steps**: Deploy K8s infrastructure, integrate Kafka producer

---

**Last Updated**: 2025-12-30  
**Version**: 4.0.0 Hybrid MLOps Edition

