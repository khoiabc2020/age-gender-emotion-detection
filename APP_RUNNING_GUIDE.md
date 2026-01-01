# 🚀 HƯỚNG DẪN CHẠY APP - SMART RETAIL AI

**Version**: 1.0  
**Date**: 2026-01-02

---

## 📋 YÊU CẦU HỆ THỐNG

### Backend API
- Python 3.11+
- PostgreSQL 14+
- Redis (optional)

### Dashboard
- Node.js 18+
- npm hoặc yarn

### Edge AI App
- Python 3.11+
- Camera (USB/RTSP) hoặc video file
- GPU (optional, khuyến nghị cho real-time)

---

## ⚡ CÁCH CHẠY NHANH NHẤT

### Option 1: Docker (Recommended)

```bash
# 1. Clone project
git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git
cd age-gender-emotion-detection

# 2. Setup environment
cp .env.example .env
# Edit .env với thông tin của bạn

# 3. Start all services
docker-compose up -d

# 4. Truy cập:
# - Dashboard: http://localhost:3000
# - API Docs: http://localhost:8000/docs
# - Login: admin / admin123
```

### Option 2: Manual (Windows)

```bash
# Chạy script tự động
START_PROJECT.bat

# Chọn option:
# 1 - Chạy Backend API
# 2 - Chạy Dashboard
# 3 - Chạy Edge App
# 4 - Chạy tất cả
```

---

## 📱 CHI TIẾT TỪNG MODULE

### 1️⃣ Backend API (FastAPI)

```bash
cd backend_api

# Cài đặt dependencies
pip install -r requirements.txt

# Setup database
# Tạo database 'smart_retail' trong PostgreSQL
createdb smart_retail

# Chạy migrations (nếu có)
# alembic upgrade head

# Chạy server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Truy cập: http://localhost:8000/docs
```

**API Endpoints:**
- `POST /auth/login` - Đăng nhập
- `GET /analytics/stats` - Thống kê
- `GET /ads/performance` - Hiệu suất quảng cáo
- `WS /ws/analytics` - Real-time updates

---

### 2️⃣ Dashboard (React + Vite)

```bash
cd dashboard

# Cài đặt dependencies
npm install

# Chạy development server
npm run dev

# Build production
npm run build

# Preview production build
npm run preview

# Truy cập: http://localhost:3000
```

**Features:**
- 📊 Analytics Dashboard - Thống kê real-time
- 👥 Demographics - Phân tích nhân khẩu học
- 🎯 Ads Management - Quản lý quảng cáo
- 🤖 AI Agent - Chat với AI về data
- ⚙️ Settings - Cấu hình hệ thống

---

### 3️⃣ Edge AI App (Computer Vision)

```bash
cd ai_edge_app

# Cài đặt dependencies
pip install -r requirements.txt

# Download model (nếu chưa có)
# Copy model từ training_experiments/checkpoints/production/best_model.pth
# -> ai_edge_app/models/multitask_model.onnx

# Chạy với camera
python main.py --camera 0

# Chạy với video file
python main.py --video path/to/video.mp4

# Chạy với RTSP stream
python main.py --rtsp rtsp://camera-ip/stream
```

**Tính năng:**
- 👤 Face Detection & Tracking
- 🎭 Emotion Recognition (7 emotions)
- 👨👩 Gender Recognition
- 🎂 Age Estimation
- 🎯 Personalized Ads Recommendation
- 📊 Real-time Analytics
- 🔄 MQTT Publishing to Backend

---

## 🔧 CẤU HÌNH

### Backend API (.env)
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/smart_retail

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# MQTT (optional)
MQTT_BROKER=localhost
MQTT_PORT=1883
```

### Dashboard (.env)
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### Edge App (configs/camera_config.json)
```json
{
  "camera_id": 0,
  "resolution": [1280, 720],
  "fps": 30,
  "model_path": "models/multitask_model.onnx",
  "mqtt_broker": "localhost",
  "mqtt_port": 1883
}
```

---

## 🎯 DEMO WORKFLOW

### Bước 1: Start Backend
```bash
cd backend_api
uvicorn app.main:app --reload
```
✅ API running at http://localhost:8000

### Bước 2: Start Dashboard
```bash
cd dashboard
npm run dev
```
✅ Dashboard at http://localhost:3000
✅ Login với `admin` / `admin123`

### Bước 3: Start Edge App
```bash
cd ai_edge_app
python main.py --camera 0
```
✅ Camera window hiện ra
✅ Nhận diện face, age, gender, emotion
✅ Hiển thị ads phù hợp

### Bước 4: Xem Analytics
- Mở Dashboard: http://localhost:3000
- Vào tab "Analytics"
- Xem real-time stats, charts
- Demographics breakdown
- Ads performance

---

## 🐛 TROUBLESHOOTING

### Backend không start?
```bash
# Check PostgreSQL running
pg_isready

# Check port 8000
netstat -an | findstr 8000

# Xem logs
tail -f logs/backend.log
```

### Dashboard không connect?
```bash
# Check .env VITE_API_URL
cat dashboard/.env

# Check CORS in backend
# backend_api/app/core/config.py
```

### Edge App không detect?
```bash
# Check model tồn tại
ls ai_edge_app/models/*.onnx

# Check camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Xem logs
tail -f ai_edge_app/logs/edge_app.log
```

---

## 📹 PREVIEW APP

### Screenshots:
```
├── docs/screenshots/
│   ├── dashboard.png       # Main dashboard
│   ├── analytics.png       # Analytics page
│   ├── edge_app.png        # Edge app running
│   └── demo.gif            # Full workflow demo
```

### Video Demo:
- Record màn hình với OBS Studio
- Hoặc dùng Windows Game Bar (Win + G)
- Export video demo

---

## 🎬 QUICK DEMO SCRIPT

1. **Start Backend**
   ```
   cd backend_api && uvicorn app.main:app
   ```

2. **Start Dashboard**
   ```
   cd dashboard && npm run dev
   ```

3. **Login Dashboard**
   - Open http://localhost:3000
   - Login: admin / admin123

4. **Start Edge App**
   ```
   cd ai_edge_app && python main.py --camera 0
   ```

5. **Show Detection**
   - Đứng trước camera
   - App nhận diện: age, gender, emotion
   - Hiển thị quảng cáo phù hợp

6. **Show Analytics**
   - Switch sang Dashboard
   - Real-time charts update
   - Demographics analysis
   - Ads performance

---

## ✅ SUCCESS CRITERIA

- ✅ Backend API: Swagger docs at /docs
- ✅ Dashboard: Login successful, charts loading
- ✅ Edge App: Face detection working, ads showing
- ✅ Real-time: Data flowing to dashboard
- ✅ MQTT: Messages publishing (optional)

---

## 📞 SUPPORT

**Issues?** Check:
1. Logs trong `logs/` folder
2. Browser console (F12)
3. Terminal output

**GitHub Issues**: https://github.com/khoiabc2020/age-gender-emotion-detection/issues

---

**Chúc may mắn với demo!** 🚀
