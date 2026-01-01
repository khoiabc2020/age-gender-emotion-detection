@echo off
chcp 65001 >nul
cls
echo ======================================================
echo   SMART RETAIL AI - START ALL SERVICES
echo ======================================================
echo.
echo This will start 3 services in separate windows:
echo   1. Backend API (http://localhost:8000)
echo   2. Dashboard (http://localhost:3000)
echo   3. Edge AI App (Camera)
echo.
echo Press any key to continue...
pause >nul

REM Get script directory and move to project root
cd /d "%~dp0\.."

echo.
echo [1/3] Starting Backend API...
start "Backend API" cmd /k "cd /d "%~dp0" && run_backend.bat"
timeout /t 2 >nul

echo [2/3] Starting Dashboard...
start "Dashboard" cmd /k "cd /d "%~dp0" && run_frontend.bat"
timeout /t 2 >nul

echo [3/3] Starting Edge AI App...
start "Edge AI" cmd /k "cd /d "%~dp0" && run_edge.bat"

cls
echo ======================================================
echo   ALL SERVICES STARTED!
echo ======================================================
echo.
echo Services are running in separate windows:
echo.
echo   Backend API:  http://localhost:8000/docs
echo   Dashboard:    http://localhost:3000
echo   Edge AI:      Check camera window
echo.
echo Login Credentials:
echo   Username: admin
echo   Password: admin123
echo.
echo To stop services:
echo   - Close each window, or
echo   - Press CTRL+C in each window
echo.
echo ======================================================
echo.
echo Press any key to close this window...
pause >nul
