# 🚀 HƯỚNG DẪN CHẠY LOCALHOST - SMART RETAIL AI

**Version**: 4.0.0 Hybrid MLOps Edition  
**Last Updated**: 2025-12-30

---

## ⚡ CÁCH NHANH NHẤT (1 Click)

### Sử dụng Script Tự Động

```bash
# Chạy script chính
START_PROJECT.bat
```

Chọn option:
- **1**: Training Test (kiểm tra training)
- **2**: Backend API (chạy server)
- **3**: Frontend Dashboard (chạy giao diện)
- **4**: Tất cả (Backend + Frontend)

---

## 📋 YÊU CẦU HỆ THỐNG

### 1. Python 3.10+
```bash
python --version
# Phải >= 3.10
```

### 2. Node.js 18+
```bash
node --version
# Phải >= 18.0.0
```

### 3. PostgreSQL (Tùy chọn)
- Có thể dùng SQLite (tự động)
- Hoặc cài PostgreSQL: https://www.postgresql.org/download/

### 4. Git (Để clone project)
```bash
git --version
```

---

## 🚀 QUICK START (3 Bước)

### Bước 1: Training Test
```bash
run_training_test.bat
```

### Bước 2: Backend API
```bash
run_backend.bat
```
**Truy cập**: http://localhost:8000/docs

### Bước 3: Frontend Dashboard
```bash
run_frontend.bat
```
**Truy cập**: http://localhost:3000  
**Login**: admin / admin123

---

## 📋 CHI TIẾT TỪNG BƯỚC

### 1️⃣ Training Test

```bash
run_training_test.bat
```

**Mục đích**: Kiểm tra training pipeline hoạt động đúng  
**Thời gian**: ~1-2 phút  
**Kết quả**: Test pass/fail

---

### 2️⃣ Backend API

#### Cách 1: Chạy trực tiếp
```bash
run_backend.bat
```

#### Cách 2: Chạy thủ công
```bash
cd backend_api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Truy cập**:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

**Default Login**:
- Username: `admin`
- Password: `admin123`

---

### 3️⃣ Frontend Dashboard

#### Cách 1: Chạy trực tiếp
```bash
run_frontend.bat
```

#### Cách 2: Chạy thủ công
```bash
cd dashboard
npm install
npm run dev
```

**Truy cập**: http://localhost:3000  
**Login**: admin / admin123

---

## 🐳 DOCKER (Production)

### Chạy với Docker Compose

```bash
# Chạy tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng services
docker-compose down
```

**Truy cập**:
- Dashboard: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🔧 CẤU HÌNH

### Backend Environment Variables

Tạo file `backend_api/.env`:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/retail_analytics
SECRET_KEY=your-secret-key-change-in-production
DEBUG=true
MQTT_BROKER=localhost
MQTT_PORT=1883
GOOGLE_AI_API_KEY=your-google-ai-key
OPENAI_API_KEY=your-openai-key
AI_PROVIDER=google_ai
```

### Frontend Environment Variables

Tạo file `dashboard/.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 🧪 TESTING

### Test Backend
```bash
python test_backend.py
```

### Test Frontend
```bash
python test_frontend.py
```

### Test System (All)
```bash
python test_system.py
```

---

## 🐛 TROUBLESHOOTING

### Backend không chạy được

1. **Kiểm tra Python version**:
   ```bash
   python --version  # Phải >= 3.10
   ```

2. **Kiểm tra dependencies**:
   ```bash
   cd backend_api
   pip install -r requirements.txt
   ```

3. **Kiểm tra port 8000**:
   ```bash
   netstat -ano | findstr :8000
   ```

### Frontend không chạy được

1. **Kiểm tra Node.js**:
   ```bash
   node --version  # Phải >= 18
   ```

2. **Xóa và cài lại dependencies**:
   ```bash
   cd dashboard
   rmdir /s /q node_modules
   del package-lock.json
   npm install
   ```

3. **Kiểm tra port 3000**:
   ```bash
   netstat -ano | findstr :3000
   ```

### Database connection error

1. **Dùng SQLite (tự động)**:
   - Backend sẽ tự tạo SQLite nếu không có PostgreSQL

2. **Hoặc setup PostgreSQL**:
   ```bash
   # Cài PostgreSQL và tạo database
   createdb retail_analytics
   ```

---

## 📚 TÀI LIỆU THAM KHẢO

- [README.md](README.md) - Tổng quan dự án
- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - Index tất cả tài liệu
- [HYBRID_MLOPS_ROADMAP.md](HYBRID_MLOPS_ROADMAP.md) - Roadmap mới
- [CI_CD_DOCUMENTATION.md](CI_CD_DOCUMENTATION.md) - CI/CD guide

---

**Status**: ✅ Complete  
**Last Updated**: 2025-12-30
