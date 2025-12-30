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

**Xem file:** [`HUONG_DAN_CHAY_LOCALHOST.md`](HUONG_DAN_CHAY_LOCALHOST.md) - Hướng dẫn đầy đủ cách chạy localhost

**Học tập & Sử dụng:** [`HUONG_DAN_HOC_TAP_VA_SU_DUNG.md`](HUONG_DAN_HOC_TAP_VA_SU_DUNG.md) - **Hướng dẫn học tập & sử dụng từ A đến Z** ⭐⭐⭐

**GitHub & Colab:** [`docs/GITHUB_AND_COLAB_GUIDE.md`](docs/GITHUB_AND_COLAB_GUIDE.md) - **Hướng dẫn upload GitHub và train trên Colab** ⭐⭐⭐

**Tài liệu tổng hợp:** [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) - Tài liệu đầy đủ về dự án

### 🐳 Docker (Production)

```bash
docker-compose up -d
```

Xem chi tiết trong [`HUONG_DAN_CHAY_LOCALHOST.md`](HUONG_DAN_CHAY_LOCALHOST.md)

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

### ⭐ Hướng Dẫn Quan Trọng
- **[HUONG_DAN_CHAY_LOCALHOST.md](HUONG_DAN_CHAY_LOCALHOST.md)** - Hướng dẫn chạy localhost chi tiết ⭐⭐⭐
- **[HUONG_DAN_HOC_TAP_VA_SU_DUNG.md](HUONG_DAN_HOC_TAP_VA_SU_DUNG.md)** - Hướng dẫn học tập & sử dụng từ A đến Z ⭐⭐⭐
- **[docs/GITHUB_AND_COLAB_GUIDE.md](docs/GITHUB_AND_COLAB_GUIDE.md)** - Hướng dẫn GitHub & Colab ⭐⭐⭐

### 📖 Tài Liệu Chính
- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - Index tất cả tài liệu
- [docs/PROJECT_DETAILS.md](docs/PROJECT_DETAILS.md) - Chi tiết dự án
- [docs/ROADMAP.md](docs/ROADMAP.md) - Roadmap phát triển
- [docs/SETUP.md](docs/SETUP.md) - Hướng dẫn setup

### 🎓 Training
- [training_experiments/README.md](training_experiments/README.md) - Training guide
- [training_experiments/AUTO_TRAINING_GUIDE.md](training_experiments/AUTO_TRAINING_GUIDE.md) - Auto training
- [training_experiments/DATASETS_INFO.md](training_experiments/DATASETS_INFO.md) - Datasets info

### 🚀 Edge App
- [ai_edge_app/README.md](ai_edge_app/README.md) - Edge app docs
- [ai_edge_app/ULTIMATE_ROADMAP.md](ai_edge_app/ULTIMATE_ROADMAP.md) - Roadmap

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

## 🆕 HYBRID MLOPS ROADMAP

Dự án đã được nâng cấp lên kiến trúc **Hybrid MLOps & Edge Ultra**:

- ☸️ **Kubernetes Infrastructure** (MinIO, Kafka, Elasticsearch)
- ⚡ **Spark Streaming** cho real-time analytics
- 🤖 **Kubeflow** cho automated ML pipelines
- 🚀 **KServe** cho model serving
- 📡 **OTA Updates** cho edge devices

Xem chi tiết: [HYBRID_MLOPS_ROADMAP.md](HYBRID_MLOPS_ROADMAP.md)
