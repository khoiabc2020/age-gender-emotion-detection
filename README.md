# 🚀 Smart Retail AI - Ultimate Edition

**Hệ thống Nhận diện Khách hàng & Đề xuất Quảng cáo Cá nhân hóa sử dụng Deep Learning và Edge Computing**

Phiên bản: Ultimate Edition v1.0 (6 Phases - In Development)

## 📋 Mô tả

Hệ thống Smart Retail Analytics là một giải pháp hoàn chỉnh từ Edge đến Cloud, sử dụng Deep Learning để:
- Nhận diện thuộc tính nhân khẩu học (Tuổi, Giới tính) và cảm xúc realtime
- Đề xuất quảng cáo động dựa trên đặc điểm khách hàng
- Phân tích hành vi người tiêu dùng qua Dashboard
- **AI Agent với Google AI và ChatGPT** (Giai đoạn 6)

## 🏗️ Kiến trúc Hệ thống

```
Edge Layer (Camera) → MQTT → Cloud Layer (Backend) → Database
                                    ↓
                              Dashboard (React)
                                    ↓
                              AI Agent (Gemini/ChatGPT)
```

## 🚀 Quick Start

### ⚡ Cách Nhanh Nhất

```bash
# Chạy script chính (Windows)
START_PROJECT.bat

# Chọn option 4 để chạy tất cả (Backend + Frontend)
```

**Truy cập:**
- Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Login: `admin` / `admin123`

### 📚 Hướng Dẫn Chi Tiết

**📖 Project Documentation:** [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) - Tài liệu đầy đủ về dự án ⭐⭐⭐⭐

**🤝 Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) - Hướng dẫn đóng góp cho dự án

### 🐳 Docker (Production)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🔑 Authentication

**Default Login:**
- Username: `admin`
- Password: `admin123`

⚠️ **Thay đổi mật khẩu trong production!**

## 🤖 AI Agent Setup

### 1. Get API Keys

**Google AI (Gemini):**
- Visit: https://makersuite.google.com/app/apikey
- Create API key
- Add to `.env`: `GOOGLE_AI_API_KEY=your-key`

**ChatGPT:**
- Visit: https://platform.openai.com/api-keys
- Create API key
- Add to `.env`: `OPENAI_API_KEY=your-key`

### 2. Configure

In `.env`:
```env
AI_PROVIDER=google_ai  # or chatgpt, or both
```

### 3. Use

1. Login to Dashboard
2. Go to Settings → AI Agent Configuration
3. Enter API keys
4. Go to AI Agent page
5. Start chatting!

## 📁 Cấu trúc Project

```
Smart-Retail-Ads/
├── ai_edge_app/          # Edge AI Application
├── backend_api/          # FastAPI Backend
├── dashboard/            # React Dashboard
├── database/             # Database scripts
├── training_experiments/ # Model training
├── mqtt/                 # MQTT config
├── docker-compose.yml    # Docker setup
└── .env.example          # Environment template
```

## 🎯 Features

### Giai đoạn 1-2: AI Core & Edge App
- ⚡ Multi-task Learning Model
- ⚡ Face Detection & Tracking
- ⚡ Real-time Analytics
- ⚡ Advertisement Engine

### Giai đoạn 3-4: Backend & Dashboard
- ⚡ RESTful API
- ⚡ WebSocket Support
- ⚡ Beautiful Dashboard
- ⚡ Real-time Updates

### Giai đoạn 5: Docker
- ⚡ Complete Docker Setup
- ⚡ Production Ready
- ⚡ Health Checks

### Giai đoạn 6: AI Agent
- ⚡ Google AI Integration
- ⚡ ChatGPT Integration
- ⚡ Chat Interface
- ⚡ Automated Reports

## 📊 API Endpoints

### Analytics
- `POST /api/v1/analytics/interactions` - Create interaction
- `GET /api/v1/analytics/stats` - Get statistics
- `GET /api/v1/analytics/age-by-hour` - Age distribution
- `GET /api/v1/analytics/emotion-distribution` - Emotion stats

### AI Agent
- `POST /api/v1/ai/analyze` - Analyze data
- `POST /api/v1/ai/chat` - Chat with AI
- `POST /api/v1/ai/generate-report` - Generate report
- `GET /api/v1/ai/status` - Check status

### Authentication
- `POST /api/v1/auth/login` - Login
- `GET /api/v1/auth/me` - Get user info

## 🛠️ Development

### Code Quality
- Type hints
- Error handling
- Logging
- Documentation

### Testing
```bash
# Backend tests
cd backend_api
pytest

# Frontend tests
cd dashboard
npm test
```

## 📚 Documentation

### ⭐ Core Documentation
- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Complete project documentation ⭐⭐⭐⭐
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

### 📖 Technical Guides
- [docs/PROJECT_DETAILS.md](docs/PROJECT_DETAILS.md) - Technical project details
- [docs/SETUP.md](docs/SETUP.md) - Environment setup guide
- [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) - Optimization report

### 🚀 Development & Deployment
- [docs/ROADMAP.md](docs/ROADMAP.md) - Development roadmap
- [docs/MLOPS_ROADMAP.md](docs/MLOPS_ROADMAP.md) - Hybrid MLOps roadmap
- [docs/PRODUCTION_ROADMAP.md](docs/PRODUCTION_ROADMAP.md) - Production deployment roadmap
- [docs/CI_CD.md](docs/CI_CD.md) - CI/CD pipeline guide

### 🔒 Security
- [docs/SECURITY.md](docs/SECURITY.md) - Security best practices
- [docs/GIT_GUIDE.md](docs/GIT_GUIDE.md) - Git workflow and commit guidelines

### 🎓 Training & AI
- [training_experiments/README.md](training_experiments/README.md) - Training guide
- [training_experiments/POST_TRAINING_WORKFLOW.md](training_experiments/POST_TRAINING_WORKFLOW.md) - Complete post-training workflow
- [training_experiments/TRAINING_VERSIONS_COMPARISON.md](training_experiments/TRAINING_VERSIONS_COMPARISON.md) - Training version comparison
- [training_experiments/notebooks/kaggle_4datasets_training.ipynb](training_experiments/notebooks/kaggle_4datasets_training.ipynb) - Main training notebook (80%+ target)

### 🚀 Edge Computing
- [ai_edge_app/README.md](ai_edge_app/README.md) - Edge application documentation

## 🔒 Security

- JWT Authentication
- Password Hashing
- CORS Configuration
- Input Validation
- SQL Injection Prevention

## 📝 License

[Your License Here]

## 👥 Contributors

[Your Name/Team]

---

**Version:** 4.0.0 Hybrid MLOps Edition  
**Status:** 🚧 In Active Development  
**Last Updated:** 2025-12-30

## 🆕 HYBRID MLOPS & PRODUCTION READY

Dự án đã được nâng cấp lên kiến trúc **Hybrid MLOps & Edge Ultra**:

- ☸️ **Kubernetes Infrastructure** (MinIO, Kafka, Elasticsearch)
- ⚡ **Spark Streaming** cho real-time analytics
- 🤖 **Kubeflow** cho automated ML pipelines
- 🚀 **KServe** cho model serving
- 📡 **OTA Updates** cho edge devices

**Xem chi tiết:**
- [docs/MLOPS_ROADMAP.md](docs/MLOPS_ROADMAP.md) - Hybrid MLOps roadmap
- [docs/PRODUCTION_ROADMAP.md](docs/PRODUCTION_ROADMAP.md) - Roadmap to production ⭐ NEW
