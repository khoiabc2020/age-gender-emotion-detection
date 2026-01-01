@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:MAIN_MENU
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     SMART RETAIL ANALYTICS - CONTROL CENTER               ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo  [1] ⚡ QUICK START      - Start Backend + Frontend
echo  [2] 🚀 Run All         - Start All Services
echo  [3] 🔧 Run Backend     - API only
echo  [4] 🌐 Run Frontend    - Dashboard only
echo  [5] 🤖 Run Edge AI     - Edge App only
echo.
echo  [6] 📦 Install All     - Install all dependencies
echo  [7] 🔍 Check Status    - Check installed packages
echo.
echo  [8] 📖 Help           - Documentation
echo  [0] ❌ Exit
echo.
echo ════════════════════════════════════════════════════════════
set /p choice="Enter your choice [0-8]: "

if "%choice%"=="1" goto QUICK_START
if "%choice%"=="2" goto RUN_ALL
if "%choice%"=="3" goto RUN_BACKEND
if "%choice%"=="4" goto RUN_FRONTEND
if "%choice%"=="5" goto RUN_EDGE
if "%choice%"=="6" goto INSTALL_ALL
if "%choice%"=="7" goto CHECK_STATUS
if "%choice%"=="8" goto HELP
if "%choice%"=="0" goto EXIT
echo Invalid choice! Press any key to try again...
pause >nul
goto MAIN_MENU

REM ============================================================
REM QUICK START - Backend + Frontend
REM ============================================================
:QUICK_START
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     QUICK START - Backend + Frontend                      ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Starting Backend API on http://localhost:8000
echo Starting Dashboard on http://localhost:3000
echo.
echo Two new windows will open. Wait 10-20 seconds...
echo.
pause

cd /d "%~dp0"
start "Backend API" cmd /k "cd /d "%~dp0backend_api" && echo Starting Backend API... && python -m app.main"
timeout /t 5 /nobreak >nul
start "Dashboard" cmd /k "cd /d "%~dp0dashboard" && echo Starting Dashboard... && npm run dev"

echo.
echo Services are starting...
echo.
echo Press any key to open browser...
pause >nul
timeout /t 10 /nobreak >nul
start http://localhost:3000

echo.
echo Services running! Close terminal windows to stop.
echo.
pause
goto MAIN_MENU

REM ============================================================
REM RUN ALL SERVICES
REM ============================================================
:RUN_ALL
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     STARTING ALL SERVICES                                  ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Starting:
echo   1. Backend API     (http://localhost:8000)
echo   2. Dashboard       (http://localhost:3000)
echo   3. Edge AI App     (Console)
echo.
echo Three new windows will open...
echo.
pause

cd /d "%~dp0"
start "Backend API" cmd /k "cd /d "%~dp0backend_api" && python -m app.main"
timeout /t 3 /nobreak >nul
start "Dashboard" cmd /k "cd /d "%~dp0dashboard" && npm run dev"
timeout /t 3 /nobreak >nul
start "Edge AI App" cmd /k "cd /d "%~dp0ai_edge_app" && python main.py"

echo.
echo All services started!
echo Close windows to stop services.
echo.
pause
goto MAIN_MENU

REM ============================================================
REM RUN BACKEND ONLY
REM ============================================================
:RUN_BACKEND
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     STARTING BACKEND API                                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0backend_api"
echo Starting on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
python -m app.main
pause
goto MAIN_MENU

REM ============================================================
REM RUN FRONTEND ONLY
REM ============================================================
:RUN_FRONTEND
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     STARTING DASHBOARD                                     ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0dashboard"
echo Starting on http://localhost:3000
echo.
call npm run dev
pause
goto MAIN_MENU

REM ============================================================
REM RUN EDGE AI ONLY
REM ============================================================
:RUN_EDGE
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     STARTING EDGE AI APP                                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0ai_edge_app"
python main.py
pause
goto MAIN_MENU

REM ============================================================
REM INSTALL ALL DEPENDENCIES
REM ============================================================
:INSTALL_ALL
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     INSTALLING ALL DEPENDENCIES                            ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo This will install:
echo   - Backend API (FastAPI, Uvicorn, SQLAlchemy...)
echo   - Dashboard (React, Ant Design, Vite...)
echo   - Edge AI (OpenCV, ONNX, NumPy...)
echo.
echo Checking Python version...
python --version
python -c "import sys; v=sys.version_info; print(f'Detected: Python {v.major}.{v.minor}'); exit(0 if v.major == 3 and v.minor <= 12 else 1)" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ════════════════════════════════════════════════════════════
    echo WARNING: Python 3.13+ detected!
    echo ════════════════════════════════════════════════════════════
    echo.
    echo ONNX Runtime may not work. Recommended: Python 3.12
    echo See PYTHON_VERSION_FIX.md for details.
    echo.
    echo Backend + Frontend will still work!
    echo ════════════════════════════════════════════════════════════
    echo.
    pause
)
echo.
echo This may take 5-10 minutes...
echo.
pause

cd /d "%~dp0"

echo.
echo [1/3] Installing Backend API dependencies...
cd backend_api
if not exist ".env" (
    echo Creating .env file...
    (
        echo # Backend Configuration
        echo SECRET_KEY=your-secret-key-change-in-production-min-32-chars
        echo ALGORITHM=HS256
        echo ACCESS_TOKEN_EXPIRE_MINUTES=30
        echo CORS_ORIGINS=["http://localhost:3000"]
        echo DEBUG=True
    ) > .env
)
pip install -r requirements.txt --upgrade --disable-pip-version-check -q
echo Backend dependencies installed!
cd ..

echo.
echo [2/3] Installing Dashboard dependencies...
cd dashboard
call npm install --silent --legacy-peer-deps 2>nul
if %errorlevel% neq 0 (
    call npm install --legacy-peer-deps
)
echo Dashboard dependencies installed!
cd ..

echo.
echo [3/3] Installing Edge AI dependencies...
cd ai_edge_app
pip install opencv-python numpy Pillow qrcode requests paho-mqtt python-dotenv --upgrade --disable-pip-version-check -q
pip install onnxruntime --upgrade --disable-pip-version-check -q 2>nul
if %errorlevel% neq 0 (
    echo.
    echo WARNING: ONNX Runtime failed. See PYTHON_VERSION_FIX.md
)
echo Edge AI dependencies installed!
cd ..

echo.
echo ════════════════════════════════════════════════════════════
echo     INSTALLATION COMPLETE!
echo ════════════════════════════════════════════════════════════
echo.
echo You can now use:
echo   [1] Quick Start    - Backend + Frontend
echo   [2] Run All        - All services
echo   [7] Check Status   - Verify installation
echo.
pause
goto MAIN_MENU

REM ============================================================
REM CHECK STATUS
REM ============================================================
:CHECK_STATUS
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     CHECKING DEPENDENCIES STATUS                           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/3] Backend API
cd backend_api
python -c "import fastapi; print('  ✓ FastAPI:', fastapi.__version__)" 2>nul || echo   ✗ FastAPI: NOT INSTALLED
python -c "import uvicorn; print('  ✓ Uvicorn:', uvicorn.__version__)" 2>nul || echo   ✗ Uvicorn: NOT INSTALLED
python -c "import sqlalchemy; print('  ✓ SQLAlchemy:', sqlalchemy.__version__)" 2>nul || echo   ✗ SQLAlchemy: NOT INSTALLED
cd ..

echo.
echo [2/3] Dashboard
cd dashboard
if exist "node_modules\" (
    echo   ✓ Node modules: INSTALLED
) else (
    echo   ✗ Node modules: NOT INSTALLED
)
cd ..

echo.
echo [3/3] Edge AI App
cd ai_edge_app
python -c "import cv2; print('  ✓ OpenCV:', cv2.__version__)" 2>nul || echo   ✗ OpenCV: NOT INSTALLED
python -c "import onnxruntime; print('  ✓ ONNX Runtime:', onnxruntime.__version__)" 2>nul || echo   ✗ ONNX Runtime: NOT INSTALLED
python -c "import numpy; print('  ✓ NumPy:', numpy.__version__)" 2>nul || echo   ✗ NumPy: NOT INSTALLED
cd ..

echo.
echo ════════════════════════════════════════════════════════════
echo If any dependencies are missing, use option [6] Install All
echo ════════════════════════════════════════════════════════════
echo.
pause
goto MAIN_MENU

REM ============================================================
REM HELP
REM ============================================================
:HELP
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     DOCUMENTATION & HELP                                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo 📖 Main Documentation:
echo   - README.md               - Project overview
echo   - QUICKSTART.md           - Quick start guide
echo   - PROJECT_DOCUMENTATION.md - Full documentation
echo.
echo 🔧 Setup Guides:
echo   - PYTHON_VERSION_FIX.md   - Python compatibility
echo   - docs/SETUP.md           - Detailed setup
echo.
echo 🎯 Frontend:
echo   - dashboard/FRONTEND_STATUS.md - Frontend details
echo   - dashboard/README.md          - Dashboard guide
echo.
echo 🤖 Training:
echo   - training_experiments/README.md - Training guide
echo   - training_experiments/TRAIN_LOCAL_GUIDE.md
echo.
echo 🌐 Access URLs:
echo   - Dashboard:  http://localhost:3000
echo   - API Docs:   http://localhost:8000/docs
echo   - API:        http://localhost:8000
echo.
echo 👤 Default Login:
echo   - Username: admin
echo   - Password: admin123
echo.
echo 📝 Common Tasks:
echo   1. First time: [6] Install All
echo   2. Daily use: [1] Quick Start
echo   3. Full system: [2] Run All
echo.
echo ════════════════════════════════════════════════════════════
pause
goto MAIN_MENU

REM ============================================================
REM EXIT
REM ============================================================
:EXIT
cls
echo Thank you for using Smart Retail Analytics!
echo.
timeout /t 2 /nobreak >nul
exit
