# 🚀 Smart Retail AI - Enterprise Edition

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI-Powered Customer Analytics & Personalized Advertisement System for Smart Retail**

A complete Edge-to-Cloud solution using Deep Learning for real-time customer demographics recognition, emotion analysis, and dynamic advertisement recommendations.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Real-time Face Detection** - RetinaFace & YOLO-based detection
- **Demographic Analysis** - Age, Gender recognition with 76%+ accuracy
- **Emotion Recognition** - 7-emotion classification
- **Smart Tracking** - DeepSORT & ByteTrack integration
- **Personalized Ads** - LinUCB-based recommendation engine
- **Anti-Spoofing** - Face liveness detection
- **Dwell Time Analysis** - Customer engagement tracking

### 🌟 Advanced Features
- **Real-time Dashboard** - React + Redux analytics interface
- **AI Agent** - Gemini/ChatGPT integration for data insights
- **WebSocket** - Live data streaming
- **MQTT** - Edge-to-cloud messaging
- **QR Codes** - Dynamic voucher generation
- **Generative Ads** - AI-powered ad slogans

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   Edge Layer    │  Camera + AI Processing
│  (Edge AI App)  │  - Face Detection
└────────┬────────┘  - Tracking
         │           - Classification
         │ MQTT      - Ad Selection
         ▼
┌─────────────────┐
│  Cloud Layer    │  API + Processing
│  (Backend API)  │  - Analytics
└────────┬────────┘  - Storage
         │           - AI Agent
         │ REST/WS
         ▼
┌─────────────────┐
│   Dashboard     │  Web Interface
│    (React)      │  - Visualization
└─────────────────┘  - Management
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Docker (optional)

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git
cd age-gender-emotion-detection

# Copy environment variables
cp .env.example .env

# Start all services
docker-compose up -d

# Access services
# Dashboard: http://localhost:3000
# API: http://localhost:8000/docs
# Login: admin / admin123
```

### Option 2: Manual Setup (Windows)

```bash
# Interactive menu
START.bat

# Or run directly:
run_app\run_all.bat        # All services
run_app\run_backend.bat    # Backend only
run_app\run_frontend.bat   # Frontend only
run_app\run_edge.bat       # Edge AI only
```

### Option 3: Development Setup

#### Backend API
```bash
cd backend_api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### Dashboard
```bash
cd dashboard
npm install
npm run dev
```

#### Edge AI App
```bash
cd ai_edge_app
pip install -r requirements.txt
python main.py --camera 0
```

---

## 📚 Documentation

### 📖 Main Guides
- **[APP_RUNNING_GUIDE.md](APP_RUNNING_GUIDE.md)** - Complete setup & running guide
- **[RECRUITMENT_READY.md](RECRUITMENT_READY.md)** - Demo guide for recruiters
- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Full documentation index
- **[CODE_CLEANUP_COMPLETED.md](CODE_CLEANUP_COMPLETED.md)** - Code quality report

### 🎓 Training & AI
- [training_experiments/README.md](training_experiments/README.md) - Training guide
- [training_experiments/POST_TRAINING_WORKFLOW.md](training_experiments/POST_TRAINING_WORKFLOW.md) - Post-training workflow  
- [training_experiments/TRAINING_SUCCESS_76.49.md](training_experiments/TRAINING_SUCCESS_76.49.md) - Training results
- [training_experiments/notebooks/kaggle_4datasets_training.ipynb](training_experiments/notebooks/kaggle_4datasets_training.ipynb) - Main training notebook

### 🛠️ Technical
- [docs/PROJECT_DETAILS.md](docs/PROJECT_DETAILS.md) - Technical architecture
- [docs/SETUP.md](docs/SETUP.md) - Environment setup
- [docs/SECURITY.md](docs/SECURITY.md) - Security best practices
- [docs/CI_CD.md](docs/CI_CD.md) - CI/CD pipeline

---

## 📂 Project Structure

```
smart-retail-ai/
├── ai_edge_app/              # Edge AI Application (Python + OpenCV)
│   ├── src/
│   │   ├── detectors/        # Face detection (RetinaFace, YOLO)
│   │   ├── trackers/         # Object tracking (DeepSORT, ByteTrack)
│   │   ├── classifiers/      # Attribute recognition
│   │   ├── ads_engine/       # Ad recommendation (LinUCB)
│   │   ├── core/             # Anti-spoofing, dwell time
│   │   ├── services/         # GenAI, QR, MQTT
│   │   └── ui/               # Visualization
│   ├── main.py               # Entry point
│   └── requirements.txt
│
├── backend_api/              # Cloud Backend (FastAPI)
│   ├── app/
│   │   ├── api/              # REST endpoints
│   │   ├── db/               # Database models
│   │   ├── services/         # Business logic
│   │   └── workers/          # Background tasks
│   ├── main.py
│   └── requirements.txt
│
├── dashboard/                # Web Dashboard (React + Redux)
│   ├── src/
│   │   ├── pages/            # Dashboard, Analytics, AI Agent
│   │   ├── components/       # Reusable UI components
│   │   ├── store/            # Redux state management
│   │   └── services/         # API clients
│   ├── package.json
│   └── vite.config.js
│
├── training_experiments/     # ML Training (Kaggle)
│   ├── notebooks/
│   │   └── kaggle_4datasets_training.ipynb
│   ├── checkpoints/          # Model checkpoints
│   └── scripts/              # Training utilities
│
├── docs/                     # Documentation
├── k8s/                      # Kubernetes configs
├── docker-compose.yml        # Docker orchestration
└── .env.example              # Environment template
```

---

## 🎯 Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 14+
- **Cache**: Redis
- **Message Queue**: MQTT (Mosquitto)
- **AI/ML**: PyTorch, ONNX Runtime
- **Computer Vision**: OpenCV, Pillow

### Frontend
- **Framework**: React 18
- **State Management**: Redux Toolkit
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts

### AI/ML Models
- **Face Detection**: RetinaFace, YOLO
- **Classification**: EfficientNet-B0 (ONNX)
- **Tracking**: DeepSORT, ByteTrack
- **Anti-Spoofing**: MiniFASNet
- **Recommendation**: LinUCB (Contextual Bandits)

### DevOps
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana (ready)
- **Logging**: Structured logging

---

## 🎓 ML Model Training

### Dataset
- **FER2013**: Emotion dataset (35,887 images)
- **UTKFace**: Age & Gender dataset (20,000+ images)
- **RAF-DB**: Real-world faces (15,339 images)
- **Total**: 70,000+ training images

### Training Results
- **Accuracy**: 76.49% (4-task multi-task learning)
- **Model**: EfficientNet-B0
- **Training Time**: ~8 hours on Kaggle P100 GPU
- **Model Size**: 70MB (PyTorch) → 35MB (ONNX)
- **Platform**: Kaggle Notebooks

### Training Notebook
See [kaggle_4datasets_training.ipynb](training_experiments/notebooks/kaggle_4datasets_training.ipynb)

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/smart_retail

# JWT Authentication
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# MQTT
MQTT_BROKER=localhost
MQTT_PORT=1883

# AI Services (Optional)
GOOGLE_AI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key

# Frontend
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### Camera Configuration

Edit `ai_edge_app/configs/camera_config.json`:

```json
{
  "camera": {
    "source": 0,
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "detection": {
    "type": "retinaface",
    "confidence_threshold": 0.8
  },
  "tracking": {
    "use_bytetrack": true,
    "dwell_threshold": 3.0
  }
}
```

---

## 🧪 Testing

### Backend Tests
```bash
cd backend_api
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd dashboard
npm test
npm run test:coverage
```

### E2E Tests
```bash
npm run test:e2e
```

---

## 🚢 Deployment

### Docker Production
```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

### Kubernetes
```bash
# Deploy to K8s
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# Check status
kubectl get pods -n smart-retail
```

---

## 📊 Performance

### Edge AI App
- **FPS**: 15-30 FPS (1080p, GTX 1660)
- **Latency**: <100ms per frame
- **CPU Usage**: ~40% (4 cores)
- **Memory**: ~2GB RAM

### Backend API
- **Throughput**: 1000+ req/s
- **Response Time**: <50ms (P95)
- **Concurrent Users**: 500+

### Dashboard
- **Load Time**: <2s
- **Bundle Size**: ~500KB (gzipped)
- **Lighthouse Score**: 95+

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@khoiabc2020](https://github.com/khoiabc2020)
- Project: [age-gender-emotion-detection](https://github.com/khoiabc2020/age-gender-emotion-detection)

---

## 🙏 Acknowledgments

- **Datasets**: FER2013, UTKFace, RAF-DB
- **Models**: RetinaFace, EfficientNet, YOLO
- **Frameworks**: FastAPI, React, PyTorch
- **Platforms**: Kaggle, Docker, GitHub

---

## 📞 Support

For questions or issues:
1. Check [Documentation](PROJECT_DOCUMENTATION.md)
2. Read [FAQ](docs/FAQ.md)
3. Create [GitHub Issue](https://github.com/khoiabc2020/age-gender-emotion-detection/issues)

---

## 🎬 Demo

**Live Demo**: [Coming Soon]  
**Video Demo**: [YouTube Link]  
**Slides**: [Presentation Link]

---

**⭐ Star this repo if you find it helpful!**

**Built with ❤️ for Smart Retail Innovation**
