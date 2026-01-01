@echo off
chcp 65001 >nul
echo ======================================================
echo   BACKEND API SERVER
echo ======================================================
echo.

cd backend_api

echo [1/2] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/2] Starting FastAPI server...
echo Access: http://localhost:8000/docs
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
