@echo off
chcp 65001 >nul
echo ======================================================
echo   SMART RETAIL AI - FULL SYSTEM
echo ======================================================
echo.
echo Starting all services...
echo.

REM Start Backend in new window
echo [1/3] Starting Backend API...
start "Backend API" cmd /k "cd /d "%~dp0" && run_backend.bat"
timeout /t 3 >nul

REM Start Frontend in new window  
echo [2/3] Starting Dashboard...
start "Dashboard" cmd /k "cd /d "%~dp0" && run_frontend.bat"
timeout /t 3 >nul

REM Start Edge App in new window
echo [3/3] Starting Edge AI App...
start "Edge AI" cmd /k "cd /d "%~dp0" && run_edge.bat"

echo.
echo ======================================================
echo   ALL SERVICES STARTED!
echo ======================================================
echo.
echo Backend API:  http://localhost:8000/docs
echo Dashboard:    http://localhost:3000
echo Edge AI:      Check camera window
echo.
echo Login: admin / admin123
echo.
echo Press any key to exit this window...
pause >nul
