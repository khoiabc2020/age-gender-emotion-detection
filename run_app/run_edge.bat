@echo off
chcp 65001 >nul
echo ======================================================
echo   EDGE AI APPLICATION
echo ======================================================
echo.

REM Navigate to project root
cd /d "%~dp0\.."
if not exist "ai_edge_app" (
    echo ERROR: ai_edge_app folder not found!
    pause
    exit /b 1
)

cd ai_edge_app

echo [1/2] Installing dependencies...
pip install opencv-python numpy pillow qrcode requests paho-mqtt python-dotenv onnxruntime --quiet --disable-pip-version-check

echo.
echo [2/2] Starting Edge AI App...
echo.
echo ^>^> Camera window will open
echo ^>^> Press 'q' to quit
echo.

python main.py --camera 0
