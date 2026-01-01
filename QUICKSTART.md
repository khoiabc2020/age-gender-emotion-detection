# 🚀 Quick Start Guide

## **Cách nhanh nhất để chạy project (2 bước)**

---

## 📋 **Bước 1: Clone & Install**

```bash
# Clone repository
git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git
cd age-gender-emotion-detection

# Run START.bat
START.bat
```

**Trong menu, chọn:**
```
[6] 📦 Install All - Install all dependencies
```

⏱️ **Thời gian:** 5-10 phút (chỉ lần đầu)

---

## 🚀 **Bước 2: Run Application**

```bash
# Run START.bat lại
START.bat
```

**Chọn một trong các options:**

### **Option 1: Quick Start (Recommended)** ⭐
```
[1] ⚡ Quick Start - Backend + Frontend
```
- ✅ Backend API: http://localhost:8000
- ✅ Dashboard: http://localhost:3000
- 🚀 Tự động mở browser

### **Option 2: Run All**
```
[2] 🚀 Run All - All Services
```
- ✅ Backend + Frontend + Edge AI
- 🎯 Chạy đầy đủ hệ thống

### **Option 3: Custom**
```
[3] 🔧 Backend only
[4] 🌐 Frontend only  
[5] 🤖 Edge AI only
```

---

## 🌐 **Truy cập ứng dụng**

| Service | URL | Login |
|---------|-----|-------|
| Dashboard | http://localhost:3000 | admin / admin123 |
| API Docs | http://localhost:8000/docs | - |
| API | http://localhost:8000 | - |

---

## ✅ **Kiểm tra cài đặt**

```bash
START.bat → [7] Check Status
```

**Kết quả mong đợi:**
```
[1/3] Backend API
  ✓ FastAPI: 0.104.0
  ✓ Uvicorn: 0.24.0
  ✓ SQLAlchemy: 2.0.0

[2/3] Dashboard
  ✓ Node modules: INSTALLED

[3/3] Edge AI App
  ✓ OpenCV: 4.8.0
  ✓ ONNX Runtime: 1.16.0
  ✓ NumPy: 1.24.0
```

---

## ❌ **Troubleshooting**

### **Lỗi: Python version**
```
ERROR: Could not find onnxruntime
```

**Giải pháp:**
- Dùng Python 3.12 (không phải 3.13+)
- Xem: [PYTHON_VERSION_FIX.md](PYTHON_VERSION_FIX.md)

### **Lỗi: Frontend trắng trang**
```
# Kiểm tra Backend đã chạy chưa
http://localhost:8000
```

**Giải pháp:**
- Backend phải chạy trước
- Dùng `START.bat → [1] Quick Start`

### **Lỗi: Dependencies thiếu**
```
START.bat → [7] Check Status
```

**Giải pháp:**
- Chạy lại: `START.bat → [6] Install All`

---

## 🐳 **Alternative: Docker**

```bash
# Đơn giản nhất, không cần lo Python version
docker-compose up -d

# Truy cập giống như trên
http://localhost:3000
```

---

## 📖 **Xem thêm**

- [README.md](README.md) - Chi tiết dự án
- [PYTHON_VERSION_FIX.md](PYTHON_VERSION_FIX.md) - Fix Python issues
- [dashboard/FRONTEND_STATUS.md](dashboard/FRONTEND_STATUS.md) - Frontend details
- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - Full docs

---

## 🎯 **Tóm tắt lệnh**

```bash
# Lần đầu
START.bat → [6] Install All

# Mỗi lần chạy
START.bat → [1] Quick Start

# Kiểm tra
START.bat → [7] Check Status

# Xem hướng dẫn
START.bat → [8] Help
```

---

**That's it! Enjoy coding! 🚀**
