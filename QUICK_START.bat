@echo off
chcp 65001 >nul
cls
echo ======================================================
echo   QUICK START - Smart Retail Analytics
echo ======================================================
echo.
echo This script will:
echo   1. Check dependencies
echo   2. Start Backend API (localhost:8000)
echo   3. Start Dashboard (localhost:3000)
echo.
echo Make sure you have run INSTALL_DEPENDENCIES.bat first!
echo.
pause

cd /d "%~dp0"

echo.
echo ======================================================
echo [1/2] Starting Backend API...
echo ======================================================
start "Backend API" cmd /k "cd /d "%~dp0backend_api" && echo Starting Backend API on http://localhost:8000 && python -m app.main"
timeout /t 5 >nul

echo.
echo ======================================================
echo [2/2] Starting Dashboard...
echo ======================================================
start "Dashboard" cmd /k "cd /d "%~dp0dashboard" && echo Starting Dashboard on http://localhost:3000 && npm run dev"

echo.
echo ======================================================
echo   SERVICES STARTING...
echo ======================================================
echo.
echo Backend API:  http://localhost:8000
echo Dashboard:    http://localhost:3000
echo.
echo Two new windows will open.
echo Wait 10-20 seconds for services to start.
echo.
echo Press any key to open browser...
pause >nul

timeout /t 10 >nul
start http://localhost:3000

echo.
echo Services are running in separate windows.
echo Close those windows to stop the services.
echo.
pause
