# 🚀 QUICK START GUIDE

## Cách chạy nhanh nhất

### 1. Chạy tất cả (Recommended)
```bash
run_all.bat
```
Script này sẽ tự động mở 3 cửa sổ:
- ✅ Backend API (http://localhost:8000)
- ✅ Dashboard (http://localhost:3000)
- ✅ Edge AI App (camera window)

### 2. Chạy riêng từng service

#### Backend API
```bash
run_backend.bat
```
Access: http://localhost:8000/docs

#### Dashboard (Frontend)
```bash
run_frontend.bat
```
Access: http://localhost:3000  
Login: `admin` / `admin123`

#### Edge AI App
```bash
run_edge.bat
```
Press 'q' to quit

---

## Lần đầu chạy

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ (optional for Backend)

### Install Dependencies
```bash
# Backend
cd backend_api
pip install -r requirements.txt

# Frontend
cd dashboard
npm install

# Edge AI
cd ai_edge_app
pip install -r requirements.txt
```

---

## Docker (Alternative)

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down
```

---

## Troubleshooting

### Backend không start?
- Check PostgreSQL running
- Edit `.env` file
- Run: `pip install -r backend_api/requirements.txt`

### Frontend không start?
- Check Node.js version: `node --version`
- Delete `node_modules` and reinstall: `npm install`

### Edge App không detect?
- Check camera: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- Download model: Copy from `training_experiments/checkpoints/`

---

## Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:3000 | admin / admin123 |
| Backend API | http://localhost:8000/docs | - |
| Edge AI | Camera window | - |

---

**Done! Enjoy!** 🎉
