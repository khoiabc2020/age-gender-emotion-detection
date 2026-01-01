@echo off
chcp 65001 >nul
echo ======================================================
echo   DASHBOARD (React Frontend)
echo ======================================================
echo.

cd /d "%~dp0\.."
cd dashboard

echo [1/2] Installing dependencies...
call npm install --legacy-peer-deps

echo.
echo [2/2] Starting development server...
echo Access: http://localhost:3000
echo Login: admin / admin123
echo.

call npm run dev
