# 📁 Run Scripts

This folder contains all startup scripts for the Smart Retail AI project.

## 🚀 Quick Start

### Run Everything
```bash
run_all.bat
```

### Run Individual Services
```bash
run_backend.bat    # Backend API only
run_frontend.bat   # Dashboard only
run_edge.bat       # Edge AI App only
```

### Legacy Script
```bash
START_PROJECT.bat  # Interactive menu (old version)
```

---

## 📖 What Each Script Does

### `run_all.bat` ⭐ RECOMMENDED
- Starts all 3 services in separate windows
- Backend API → http://localhost:8000
- Dashboard → http://localhost:3000
- Edge AI → Camera window

### `run_backend.bat`
- Installs Python dependencies
- Starts FastAPI server
- Access Swagger docs at /docs

### `run_frontend.bat`
- Installs Node.js dependencies
- Starts React development server
- Login: admin / admin123

### `run_edge.bat`
- Installs Python dependencies
- Starts camera-based AI processing
- Press 'q' to quit

---

## ⚡ Usage

From project root:
```bash
# Interactive launcher
START.bat

# Or direct
run_app\run_all.bat
```

From this folder:
```bash
cd run_app
run_all.bat
```

---

**Easy to use!** 🎉
