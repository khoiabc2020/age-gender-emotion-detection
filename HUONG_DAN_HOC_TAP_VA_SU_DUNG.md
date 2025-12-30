# 📚 HƯỚNG DẪN HỌC TẬP & SỬ DỤNG DỰ ÁN - SMART RETAIL AI

**Version**: Ultimate Edition v1.0  
**Last Updated**: 2025-12-30  
**Status**: ✅ Complete - Từ A đến Z

---

## ⭐ TÓM TẮT NHANH

File này hướng dẫn **ĐẦY ĐỦ** từ A đến Z:
- ✅ **Kiến thức cần học** để làm nên dự án này
- ✅ **Hướng dẫn sử dụng** chi tiết từng bước
- ✅ **Cấu trúc dự án** và workflow phát triển
- ✅ **Troubleshooting** các lỗi thường gặp

**Bắt đầu ngay**: Xem [PHẦN A: SETUP MÔI TRƯỜNG](#phần-a-setup-môi-trường)

---

---

## 📋 MỤC LỤC

1. [Kiến thức cần học](#kiến-thức-cần-học)
2. [Hướng dẫn sử dụng từ A đến Z](#hướng-dẫn-sử-dụng-từ-a-đến-z)
3. [Cấu trúc dự án chi tiết](#cấu-trúc-dự-án-chi-tiết)
4. [Workflow phát triển](#workflow-phát-triển)
5. [Troubleshooting](#troubleshooting)

---

## 🎓 KIẾN THỨC CẦN HỌC

### 1. **Python & Deep Learning Fundamentals**

#### Python Cơ bản
- ✅ **Syntax cơ bản**: Variables, Data types, Functions, Classes
- ✅ **OOP**: Inheritance, Polymorphism, Encapsulation
- ✅ **Modules & Packages**: Import, __init__.py, Package structure
- ✅ **File I/O**: Reading/Writing files, JSON, CSV
- ✅ **Error Handling**: Try-except, Custom exceptions
- ✅ **Logging**: Python logging module

**Tài liệu học:**
- Python Official Docs: https://docs.python.org/3/
- Real Python: https://realpython.com/

#### NumPy & Pandas
- ✅ **NumPy**: Arrays, Broadcasting, Operations
- ✅ **Pandas**: DataFrames, Series, Data manipulation
- ✅ **Data Cleaning**: Handling missing values, duplicates

**Tài liệu học:**
- NumPy Tutorial: https://numpy.org/doc/stable/user/quickstart.html
- Pandas Tutorial: https://pandas.pydata.org/docs/getting_started/

#### PyTorch Deep Learning
- ✅ **Tensors**: Creation, Operations, GPU support
- ✅ **Neural Networks**: nn.Module, Layers, Activation functions
- ✅ **Training Loop**: Forward pass, Backward pass, Optimizers
- ✅ **Loss Functions**: CrossEntropy, MSE, Custom losses
- ✅ **Data Loading**: Dataset, DataLoader, Transforms
- ✅ **Model Saving/Loading**: .pth files, State dict

**Tài liệu học:**
- PyTorch Official Tutorial: https://pytorch.org/tutorials/
- Deep Learning with PyTorch: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

#### Advanced PyTorch
- ✅ **Transfer Learning**: Pre-trained models, Fine-tuning
- ✅ **Multi-task Learning**: Shared backbone, Multiple heads
- ✅ **Knowledge Distillation**: Teacher-Student learning
- ✅ **Quantization**: QAT, INT8 quantization
- ✅ **Mixed Precision**: FP16 training, GradScaler
- ✅ **ONNX Export**: Model conversion, Opset versions

**Tài liệu học:**
- PyTorch Advanced: https://pytorch.org/tutorials/intermediate/
- ONNX Tutorial: https://onnx.ai/onnx/intro/

---

### 2. **Computer Vision**

#### Image Processing
- ✅ **OpenCV**: Image reading, Resizing, Color spaces
- ✅ **Image Augmentation**: Rotation, Flip, Brightness, Contrast
- ✅ **Albumentations**: Advanced augmentation library
- ✅ **Face Detection**: Haar Cascade, RetinaFace, YuNet
- ✅ **Face Recognition**: Feature extraction, Embeddings

**Tài liệu học:**
- OpenCV Tutorial: https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
- Albumentations: https://albumentations.ai/docs/

#### Deep Learning for Vision
- ✅ **CNN Architectures**: ResNet, EfficientNet, MobileNet
- ✅ **Attention Mechanisms**: SE-Block, CBAM
- ✅ **Object Detection**: YOLO, RetinaNet
- ✅ **Object Tracking**: DeepSORT, ByteTrack
- ✅ **Face Analysis**: Age, Gender, Emotion recognition

**Tài liệu học:**
- CS231n Stanford: http://cs231n.stanford.edu/
- Papers with Code: https://paperswithcode.com/

---

### 3. **Edge Computing & Optimization**

#### Model Optimization
- ✅ **Model Compression**: Pruning, Quantization
- ✅ **Knowledge Distillation**: Teacher-Student
- ✅ **Mobile Architectures**: MobileNet, MobileOne, FastViT
- ✅ **ONNX Runtime**: Inference optimization
- ✅ **TensorRT/OpenVINO**: Hardware acceleration

**Tài liệu học:**
- ONNX Runtime: https://onnxruntime.ai/docs/
- TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/

#### Edge Deployment
- ✅ **ONNX Format**: Model conversion, Opset versions
- ✅ **Edge Devices**: Jetson Nano, Raspberry Pi, Laptop
- ✅ **Performance Optimization**: FPS, Latency, Memory
- ✅ **Multi-threading**: QThread, Producer-Consumer pattern

---

### 4. **Backend Development**

#### FastAPI
- ✅ **RESTful API**: GET, POST, PUT, DELETE
- ✅ **Request/Response Models**: Pydantic models
- ✅ **Database Integration**: SQLAlchemy ORM
- ✅ **Authentication**: JWT tokens, Password hashing
- ✅ **WebSocket**: Real-time communication
- ✅ **API Documentation**: Swagger/OpenAPI

**Tài liệu học:**
- FastAPI Docs: https://fastapi.tiangolo.com/
- SQLAlchemy: https://docs.sqlalchemy.org/

#### Database
- ✅ **SQLite**: Local database, SQL queries
- ✅ **PostgreSQL**: Production database
- ✅ **SQLAlchemy**: ORM, Models, Relationships
- ✅ **TimescaleDB**: Time-series data (optional)

---

### 5. **Frontend Development**

#### React.js
- ✅ **Components**: Functional components, Hooks
- ✅ **State Management**: useState, useContext, Redux
- ✅ **Routing**: React Router
- ✅ **API Integration**: Axios, Fetch
- ✅ **Styling**: CSS, Tailwind CSS, Styled Components

**Tài liệu học:**
- React Official: https://react.dev/
- Tailwind CSS: https://tailwindcss.com/docs

#### Modern UI Libraries
- ✅ **Ant Design**: Component library
- ✅ **Recharts**: Data visualization
- ✅ **Tremor/ShadcnUI**: Modern dashboard components

---

### 6. **DevOps & MLOps**

#### Docker
- ✅ **Dockerfile**: Image building
- ✅ **Docker Compose**: Multi-container orchestration
- ✅ **Containerization**: Best practices

**Tài liệu học:**
- Docker Docs: https://docs.docker.com/

#### Kubernetes (Optional - Advanced)
- ✅ **K8s Basics**: Pods, Services, Deployments
- ✅ **Kubeflow**: ML pipelines
- ✅ **KServe**: Model serving

#### CI/CD
- ✅ **GitHub Actions**: Automated testing, deployment
- ✅ **Git Workflow**: Branching, Commits, PRs

---

### 7. **Message Queuing & Streaming**

#### MQTT
- ✅ **Pub/Sub Pattern**: Topics, Messages
- ✅ **MQTT Clients**: paho-mqtt library

#### Kafka (Optional - Advanced)
- ✅ **Event Streaming**: Producers, Consumers
- ✅ **Topics & Partitions**: Data distribution

---

### 8. **UI Development (Edge App)**

#### PyQt6
- ✅ **Widgets**: QWidget, QMainWindow, Layouts
- ✅ **Signals & Slots**: Event handling
- ✅ **QThread**: Multi-threading
- ✅ **QMediaPlayer**: Video playback
- ✅ **Custom Painting**: QPainter, QGraphics

**Tài liệu học:**
- PyQt6 Docs: https://www.riverbankcomputing.com/static/Docs/PyQt6/

#### QFluentWidgets
- ✅ **Modern UI Components**: Fluent design
- ✅ **Theming**: Dark/Light mode
- ✅ **Glassmorphism**: Acrylic effects

---

## 🚀 HƯỚNG DẪN SỬ DỤNG TỪ A ĐẾN Z

### PHẦN A: SETUP MÔI TRƯỜNG

#### A1. Cài đặt Python 3.10+

```bash
# Kiểm tra Python
python --version
# Phải >= 3.10

# Nếu chưa có, download từ: https://www.python.org/downloads/
```

#### A2. Cài đặt Node.js 18+

```bash
# Kiểm tra Node.js
node --version
# Phải >= 18.0.0

# Nếu chưa có, download từ: https://nodejs.org/
```

#### A3. Cài đặt Git

```bash
# Kiểm tra Git
git --version

# Nếu chưa có, download từ: https://git-scm.com/downloads
```

#### A4. Clone Project

```bash
git clone <repository-url>
cd "nhan dien do tuoi"
```

---

### PHẦN B: TRAINING MODEL

#### B1. Chuẩn bị Dữ liệu

```bash
cd training_experiments

# Download datasets từ Kaggle
python scripts/download_datasets.py

# Copy datasets vào project
python scripts/copy_datasets_to_project.py

# Kiểm tra datasets
python scripts/check_datasets.py
```

**Datasets cần:**
- UTKFace: 23,708 images (Age, Gender)
- FER2013: 28,709 images (Emotion)

#### B2. Setup Virtual Environment

```bash
# Tạo venv
python -m venv venv_gpu

# Activate (Windows)
venv_gpu\Scripts\activate

# Activate (Linux/Mac)
source venv_gpu/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

#### B3. Training Model

##### Option 1: Training đơn giản
```bash
python train_week2_lightweight.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

##### Option 2: Training với Knowledge Distillation
```bash
python train_week2_lightweight.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_distillation
```

##### Option 3: Training 10 lần tự động
```bash
python train_10x_automated.py
```

**Kết quả:**
- Model: `models/best_model.pth`
- ONNX: `models/best_model.onnx`
- Logs: `checkpoints/logs/`

#### B4. Convert sang ONNX

```bash
python scripts/convert_to_onnx.py \
    --model_path models/best_model.pth \
    --output_path models/multitask_efficientnet.onnx \
    --opset_version 13
```

#### B5. Copy Model vào Edge App

```bash
# Copy ONNX model
copy models\multitask_efficientnet.onnx ai_edge_app\models\
```

---

### PHẦN C: CHẠY EDGE APP

#### C1. Setup Edge App

```bash
cd ai_edge_app

# Cài dependencies
pip install -r requirements.txt
```

#### C2. Cấu hình Camera

Chỉnh sửa `configs/camera_config.json`:
```json
{
  "camera": {
    "source": 0,  // 0 = webcam, hoặc đường dẫn video
    "width": 640,
    "height": 480,
    "fps": 30
  },
  "tracking": {
    "use_bytetrack": true,
    "dwell_threshold": 3.0
  }
}
```

#### C3. Chạy Edge App

##### Option 1: OpenCV Version (Đơn giản)
```bash
python main.py
```

##### Option 2: PyQt6 Version (Modern UI)
```bash
python -m src.ui.main_window
```

**Features:**
- Real-time face detection
- Age, Gender, Emotion recognition
- Ad recommendation
- Smart tracking với ByteTrack
- Dwell time logic

---

### PHẦN D: CHẠY BACKEND API

#### D1. Setup Backend

```bash
cd backend_api

# Tạo venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài dependencies
pip install -r requirements.txt
```

#### D2. Cấu hình Database

Tạo file `.env`:
```env
DATABASE_URL=sqlite:///./retail_analytics.db
# Hoặc PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/retail_analytics

SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

#### D3. Chạy Backend

```bash
# Chạy server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Truy cập:**
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

#### D4. Test API

```bash
# Health check
curl http://localhost:8000/health

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

---

### PHẦN E: CHẠY FRONTEND DASHBOARD

#### E1. Setup Frontend

```bash
cd dashboard

# Cài dependencies
npm install
```

#### E2. Cấu hình Environment

Tạo file `.env.local`:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

#### E3. Chạy Frontend

```bash
npm run dev
```

**Truy cập:**
- Dashboard: http://localhost:3000
- Login: `admin` / `admin123`

#### E4. Sử dụng Dashboard

1. **Login**: Đăng nhập với admin/admin123
2. **Dashboard**: Xem tổng quan analytics
3. **Analytics**: Phân tích chi tiết
4. **Ads Management**: Quản lý quảng cáo
5. **Settings**: Cài đặt hệ thống
6. **AI Agent**: Chat với AI (cần API keys)

---

### PHẦN F: TÍCH HỢP AI AGENT

#### F1. Lấy API Keys

**Google AI (Gemini):**
1. Truy cập: https://makersuite.google.com/app/apikey
2. Tạo API key
3. Copy key

**ChatGPT:**
1. Truy cập: https://platform.openai.com/api-keys
2. Tạo API key
3. Copy key

#### F2. Cấu hình Backend

Thêm vào `backend_api/.env`:
```env
GOOGLE_AI_API_KEY=your-google-ai-key
OPENAI_API_KEY=your-openai-key
AI_PROVIDER=google_ai  # hoặc chatgpt, hoặc both
```

#### F3. Sử dụng AI Agent

1. Login vào Dashboard
2. Vào Settings → AI Agent Configuration
3. Nhập API keys
4. Vào AI Agent page
5. Bắt đầu chat!

---

### PHẦN G: DOCKER DEPLOYMENT

#### G1. Build Images

```bash
# Build tất cả images
docker-compose build
```

#### G2. Chạy với Docker

```bash
# Chạy tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f

# Stop
docker-compose down
```

#### G3. Kiểm tra Services

```bash
# List containers
docker-compose ps

# Health check
curl http://localhost:8000/health
```

---

## 📁 CẤU TRÚC DỰ ÁN CHI TIẾT

```
nhan dien do tuoi/
├── 📂 training_experiments/     # Model Training
│   ├── src/
│   │   ├── models/              # Model architectures
│   │   │   ├── mobileone.py    # MobileOne-S2
│   │   │   ├── network.py      # Base model
│   │   │   ├── ultimate_network.py  # Ultimate model
│   │   │   ├── knowledge_distillation.py  # KD
│   │   │   └── qat_model.py    # QAT
│   │   ├── data/
│   │   │   ├── dataset.py      # DataLoader
│   │   │   ├── preprocess.py   # Preprocessing
│   │   │   └── advanced_preprocess.py  # Advanced
│   │   └── utils/
│   │       └── logging.py      # Logging
│   ├── scripts/
│   │   ├── download_datasets.py
│   │   ├── convert_to_onnx.py
│   │   └── check_week*_requirements.py
│   ├── train_week2_lightweight.py  # Main training
│   ├── train_10x_automated.py      # Auto training
│   └── requirements.txt
│
├── 📂 ai_edge_app/              # Edge Application
│   ├── src/
│   │   ├── core/
│   │   │   ├── anti_spoofing.py    # MiniFASNet
│   │   │   ├── face_restoration.py # GFPGAN/ESPCN
│   │   │   ├── dwell_time.py       # Dwell time logic
│   │   │   └── multithreading.py   # QThread
│   │   ├── detectors/
│   │   │   └── retinaface_detector.py
│   │   ├── trackers/
│   │   │   ├── bytetrack_tracker.py
│   │   │   └── deepsort_tracker.py
│   │   ├── classifiers/
│   │   │   └── multitask_classifier.py
│   │   ├── ads_engine/
│   │   │   ├── ads_selector.py     # LinUCB
│   │   │   └── lin_ucb.py
│   │   ├── ui/
│   │   │   ├── main_window.py      # PyQt6 UI
│   │   │   ├── glassmorphism.py
│   │   │   ├── smart_overlay.py
│   │   │   ├── live_charts.py
│   │   │   └── ads_player.py
│   │   ├── database/
│   │   │   ├── models.py           # SQLAlchemy
│   │   │   ├── db_manager.py
│   │   │   └── export_manager.py   # Excel/PDF
│   │   ├── gesture/
│   │   │   └── gesture_recognizer.py  # MediaPipe
│   │   └── services/
│   │       ├── kafka_producer.py
│   │       ├── model_ota.py
│   │       └── generative_ads.py
│   ├── main.py                   # Main entry
│   ├── configs/
│   │   ├── camera_config.json
│   │   └── ads_rules.json
│   └── requirements.txt
│
├── 📂 backend_api/               # FastAPI Backend
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── models/              # Database models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── routes/              # API routes
│   │   │   ├── auth.py
│   │   │   ├── analytics.py
│   │   │   └── ai_agent.py
│   │   └── services/            # Business logic
│   ├── .env.example
│   └── requirements.txt
│
├── 📂 dashboard/                 # React Frontend
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── pages/              # Pages
│   │   ├── services/           # API services
│   │   ├── store/              # Redux store
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── 📂 docs/                      # Documentation
│   ├── ROADMAP.md
│   ├── PROJECT_DETAILS.md
│   └── SETUP.md
│
├── 📄 README.md                  # Main README
├── 📄 PROJECT_DOCUMENTATION.md   # Documentation index
├── 📄 HUONG_DAN_CHAY_LOCALHOST.md # How to run
├── 📄 HUONG_DAN_HOC_TAP_VA_SU_DUNG.md  # This file
├── 📄 START_PROJECT.bat          # Main script
└── 📄 docker-compose.yml         # Docker setup
```

---

## 🔄 WORKFLOW PHÁT TRIỂN

### 1. Development Workflow

```
1. Training Model
   ├── Prepare data
   ├── Train model
   ├── Evaluate
   ├── Convert to ONNX
   └── Copy to edge app

2. Edge App Development
   ├── Test detection
   ├── Test tracking
   ├── Test classification
   ├── Test UI
   └── Test ads engine

3. Backend Development
   ├── Design API
   ├── Implement routes
   ├── Test endpoints
   └── Update docs

4. Frontend Development
   ├── Design UI
   ├── Implement components
   ├── Connect API
   └── Test features
```

### 2. Testing Workflow

```bash
# 1. Test Training
cd training_experiments
python scripts/test_pipeline.py

# 2. Test Edge App
cd ai_edge_app
python main.py

# 3. Test Backend
cd backend_api
pytest

# 4. Test Frontend
cd dashboard
npm test
```

---

## 🐛 TROUBLESHOOTING

### Lỗi thường gặp

#### 1. Training lỗi "CUDA out of memory"
**Giải pháp:**
- Giảm batch_size
- Sử dụng gradient accumulation
- Sử dụng mixed precision

#### 2. Model không load được
**Giải pháp:**
- Kiểm tra đường dẫn model
- Kiểm tra ONNX opset version
- Kiểm tra input shape

#### 3. Camera không mở được
**Giải pháp:**
- Kiểm tra camera index (0, 1, 2...)
- Kiểm tra permissions
- Thử đường dẫn video file

#### 4. Backend không chạy
**Giải pháp:**
- Kiểm tra port 8000 có bị chiếm không
- Kiểm tra .env file
- Kiểm tra database connection

#### 5. Frontend không kết nối API
**Giải pháp:**
- Kiểm tra VITE_API_URL trong .env.local
- Kiểm tra CORS settings
- Kiểm tra backend đang chạy

---

## 📚 TÀI LIỆU THAM KHẢO

### Official Docs
- PyTorch: https://pytorch.org/docs/
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- OpenCV: https://docs.opencv.org/

### Tutorials
- PyTorch Tutorials: https://pytorch.org/tutorials/
- FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/
- React Tutorial: https://react.dev/learn

### Papers
- MobileOne: https://arxiv.org/abs/2206.04040
- ByteTrack: https://arxiv.org/abs/2110.06864
- Knowledge Distillation: https://arxiv.org/abs/1503.02531

---

## 🎯 NEXT STEPS

Sau khi học xong các kiến thức trên:

1. **Bắt đầu với Training**: Làm quen với PyTorch
2. **Phát triển Edge App**: Hiểu về Computer Vision
3. **Xây dựng Backend**: Học FastAPI
4. **Tạo Frontend**: Học React
5. **Deploy**: Học Docker

---

**Chúc bạn học tập và phát triển dự án thành công!** 🚀

