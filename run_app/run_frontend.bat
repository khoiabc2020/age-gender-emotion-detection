@echo off
chcp 65001 >nul
echo ======================================================
echo   DASHBOARD (React Frontend)
echo ======================================================
echo.

REM Navigate to project root then dashboard
cd /d "%~dp0\.."
if not exist "dashboard" (
    echo ERROR: dashboard folder not found!
    pause
    exit /b 1
)

cd dashboard

echo [1/2] Installing dependencies...
call npm install --legacy-peer-deps --silent
if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install
    echo Continuing anyway...
    echo.
)

echo.
echo [2/2] Starting development server...
echo.
echo ^>^> Dashboard: http://localhost:3000
echo ^>^> Login: admin / admin123
echo ^>^> Press CTRL+C to stop
echo.

call npm run dev
