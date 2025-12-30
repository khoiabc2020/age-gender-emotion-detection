@echo off
REM Backend API Server - FastAPI
echo ========================================
echo STARTING BACKEND API SERVER
echo ========================================
echo.

cd backend_api

REM Check if .env exists
if not exist .env (
    echo [WARNING] .env file not found, creating default...
    (
        echo DATABASE_URL=postgresql://postgres:postgres@localhost:5432/retail_analytics
        echo SECRET_KEY=your-secret-key-change-in-production
        echo DEBUG=true
        echo MQTT_BROKER=localhost
        echo MQTT_PORT=1883
        echo CORS_ORIGINS=["http://localhost:3000","http://localhost:8501"]
    ) > .env
    echo [OK] Default .env created
)

REM Check if dependencies installed
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo [INFO] Starting FastAPI server...
echo [INFO] API will be available at: http://localhost:8000
echo [INFO] API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

