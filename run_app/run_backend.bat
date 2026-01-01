@echo off
chcp 65001 >nul
echo ======================================================
echo   BACKEND API SERVER
echo ======================================================
echo.

REM Navigate to project root then backend
cd /d "%~dp0\.."
if not exist "backend_api" (
    echo ERROR: backend_api folder not found!
    pause
    exit /b 1
)

cd backend_api

REM Check if .env exists, if not create it
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
    echo .env file created!
    echo.
)

echo [1/2] Installing dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install
    echo Continuing anyway...
    echo.
)

echo.
echo [2/2] Starting FastAPI server...
echo.
echo ^>^> Backend API: http://localhost:8000/docs
echo ^>^> Press CTRL+C to stop
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
