@echo off
chcp 65001 >nul
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║     TESTING ALL 3 SERVICES TOGETHER                       ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Starting:
echo   1. Backend API     (http://localhost:8000)
echo   2. Dashboard       (http://localhost:3000)
echo   3. Edge AI App     (Console)
echo.
echo Optimizations applied:
echo   - Edge AI: Frame skipping, reduced resolution (320x240)
echo   - Frontend: Delayed initial load
echo   - Backend: Auto port selection
echo.
pause

cd /d "%~dp0"

echo.
echo [1/3] Starting Backend API...
start "Backend API" cmd /k "cd /d "%~dp0backend_api" && echo Backend API starting... && python -m app.main"
timeout /t 5 /nobreak >nul

echo.
echo [2/3] Starting Dashboard...
start "Dashboard" cmd /k "cd /d "%~dp0dashboard" && echo Dashboard starting... && npm run dev"
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Starting Edge AI App...
start "Edge AI App" cmd /k "cd /d "%~dp0ai_edge_app" && echo Edge AI App starting... && python main.py"

echo.
echo ════════════════════════════════════════════════════════════
echo   ALL SERVICES STARTED!
echo ════════════════════════════════════════════════════════════
echo.
echo Waiting 15 seconds for services to initialize...
timeout /t 15 /nobreak >nul

echo.
echo Opening browser...
start http://localhost:3000

echo.
echo ════════════════════════════════════════════════════════════
echo   SERVICES RUNNING
echo ════════════════════════════════════════════════════════════
echo.
echo Backend API:  http://localhost:8000/docs
echo Dashboard:    http://localhost:3000
echo.
echo Close the 3 windows to stop services.
echo.
pause
