# ✅ ĐÃ HOÀN TẤT - HƯỚNG DẪN CHẠY APP & CLEAN CODE

**Date**: 2026-01-02  
**Status**: Ready for recruitment review

---

## 📋 TÓM TẮT

Tôi đã tạo cho bạn:
1. ✅ **APP_RUNNING_GUIDE.md** - Hướng dẫn chạy app đầy đủ
2. ✅ **CODE_CLEANUP_PLAN.md** - Kế hoạch clean code chi tiết
3. ✅ **clean_code.py** - Script tự động clean code
4. ✅ **32 files đã xóa** - Project đã gọn gàng

---

## 🚀 CÁCH CHẠY APP NHANH NHẤT

### Option 1: Docker (Khuyến nghị)
```bash
# Clone & start
git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git
cd age-gender-emotion-detection
cp .env.example .env
docker-compose up -d

# Truy cập
Dashboard: http://localhost:3000
API: http://localhost:8000/docs
Login: admin / admin123
```

### Option 2: Windows Manual
```bash
# Chạy script
START_PROJECT.bat

# Chọn option 4 - Chạy tất cả
```

---

## 🧹 CLEAN CODE (Quan trọng cho nhà tuyển dụng!)

### Bước 1: Chạy auto-cleanup
```bash
python clean_code.py
```

Script này sẽ:
- ✅ Xóa comments "Tuần X"
- ✅ Xóa comments tiếng Việt debug
- ✅ Xóa emoji trong code
- ✅ Chuẩn hóa format

### Bước 2: Manual review (optional)
Xem `CODE_CLEANUP_PLAN.md` để review thêm.

### Bước 3: Format code
```bash
# Python
cd ai_edge_app && black . && isort .
cd backend_api && black . && isort .

# JavaScript
cd dashboard && npm run format
```

---

## 🎯 DEMO CHO NHÀ TUYỂN DỤNG

### Scenario 1: Quick Demo (5 phút)
1. **Start services** (2 phút)
   ```bash
   START_PROJECT.bat  # Option 4
   ```

2. **Show Dashboard** (1 phút)
   - Open http://localhost:3000
   - Login: admin / admin123
   - Show Analytics, Charts

3. **Start Edge App** (2 phút)
   ```bash
   cd ai_edge_app
   python main.py --camera 0
   ```
   - Đứng trước camera
   - Nhận diện face, age, gender, emotion
   - Quảng cáo hiện ra
   - Dashboard update real-time

### Scenario 2: Full Demo (15 phút)
1. **Architecture Overview** (3 phút)
   - Show docker-compose.yml
   - Explain: Edge → MQTT → Backend → Dashboard
   - 3-tier architecture

2. **Backend API** (3 phút)
   - http://localhost:8000/docs
   - Show endpoints
   - Try /analytics/stats

3. **Dashboard** (4 phút)
   - Login
   - Analytics page - Real-time stats
   - Demographics charts
   - Ads performance
   - AI Agent (chat về data)

4. **Edge AI** (5 phút)
   - Start camera
   - Face detection
   - Attribute recognition
   - Personalized ads
   - Show MQTT messages

---

## 📊 TECHNICAL HIGHLIGHTS (Nói với nhà tuyển dụng)

### 1. Architecture
- **Microservices**: Backend API, Dashboard, Edge App
- **Real-time**: WebSocket + MQTT
- **Scalable**: Docker + Kubernetes ready
- **Cloud-native**: AWS/GCP deployment ready

### 2. AI/ML Stack
- **Deep Learning**: PyTorch, ONNX Runtime
- **Models**: EfficientNet, RetinaFace, YOLO
- **Training**: Kaggle (4 datasets, 76.49% accuracy)
- **Edge Optimization**: ONNX, quantization

### 3. Full-Stack Development
- **Backend**: FastAPI, PostgreSQL, Redis
- **Frontend**: React 18, Redux Toolkit, Vite
- **Edge**: OpenCV, NumPy, threading optimization
- **DevOps**: Docker, CI/CD (GitHub Actions)

### 4. Advanced Features
- **Face Tracking**: DeepSORT, ByteTrack
- **Anti-Spoofing**: MiniFASNet
- **Dwell Time**: Customer engagement tracking
- **Recommendation**: LinUCB (reinforcement learning)
- **GenAI**: Gemini API for dynamic content

### 5. Production-Ready
- **Testing**: Pytest, Vitest
- **Logging**: Structured logging
- **Monitoring**: Prometheus + Grafana ready
- **Security**: JWT auth, CORS, SSL
- **Documentation**: Comprehensive docs

---

## 💼 CHUẨN BỊ CHO PHỎNG VẤN

### Câu hỏi thường gặp:

**Q: Làm thế nào scale hệ thống?**
A: 
- Backend: Load balancer + multiple instances
- Edge: Deploy nhiều cameras → 1 backend
- Database: PostgreSQL replication
- Cache: Redis cluster
- Kubernetes: Auto-scaling

**Q: Performance optimization?**
A:
- Edge: ONNX Runtime, multi-threading
- Backend: FastAPI async, connection pooling
- Frontend: Code splitting, lazy loading
- Database: Indexing, query optimization

**Q: Security?**
A:
- Auth: JWT tokens
- API: Rate limiting, CORS
- Data: Encryption at rest/transit
- Edge: Device authentication via MQTT

**Q: Testing strategy?**
A:
- Unit tests: Pytest (backend), Vitest (frontend)
- Integration tests: API endpoints
- E2E tests: Playwright/Cypress
- Load tests: Locust/K6

---

## 📂 PROJECT STRUCTURE (Show nhà tuyển dụng)

```
smart-retail-ai/
├── ai_edge_app/           # Edge Computing (Python + OpenCV)
│   ├── src/
│   │   ├── detectors/     # Face detection
│   │   ├── trackers/      # Object tracking
│   │   ├── classifiers/   # Attribute recognition
│   │   └── ads_engine/    # Recommendation
│   └── main.py
├── backend_api/           # Cloud Backend (FastAPI)
│   ├── app/
│   │   ├── api/           # REST endpoints
│   │   ├── db/            # Database models
│   │   └── services/      # Business logic
│   └── main.py
├── dashboard/             # Web Dashboard (React)
│   ├── src/
│   │   ├── pages/         # Dashboard, Analytics
│   │   ├── components/    # Reusable UI
│   │   └── store/         # Redux state
│   └── vite.config.js
├── training_experiments/  # ML Training (Kaggle)
│   ├── notebooks/
│   │   └── kaggle_4datasets_training.ipynb
│   └── checkpoints/
└── docker-compose.yml     # Orchestration
```

---

## ✅ CHECKLIST TRƯỚC DEMO

- [ ] Code đã clean (chạy `python clean_code.py`)
- [ ] All services start successfully
- [ ] Camera hoạt động
- [ ] Dashboard login OK
- [ ] Real-time data flow
- [ ] No errors in console/logs
- [ ] README updated
- [ ] Git pushed

---

## 🎬 RECORDING DEMO

### Tools:
- **OBS Studio** (free, professional)
- **Windows Game Bar** (Win + G)
- **Loom** (web-based)

### Script:
1. Intro (30s): "Smart Retail AI - Real-time customer analytics"
2. Architecture (1min): Show diagram
3. Dashboard (2min): Login, analytics, charts
4. Edge App (2min): Face detection, attributes, ads
5. Outro (30s): "Thank you!"

Total: ~6 minutes

---

## 📞 FINAL TIPS

1. **Confidence**: Nói về tech stack và design decisions
2. **Show, don't tell**: Demo trực tiếp > slides
3. **Handle errors**: Prepare for "what if camera fails?"
4. **Be honest**: "This would be improved by..."
5. **Future vision**: "Next steps: Kubernetes, more models..."

---

**Good luck với interview!** 🚀🎯

Nếu cần support:
- Check logs: `logs/` folder
- GitHub: https://github.com/khoiabc2020/age-gender-emotion-detection
- Issues: Create GitHub issue
