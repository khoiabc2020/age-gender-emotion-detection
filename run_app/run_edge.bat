@echo off
chcp 65001 >nul
echo ======================================================
echo   EDGE AI APPLICATION
echo ======================================================
echo.

REM Navigate to project root then ai_edge_app
cd /d "%~dp0\.."
if not exist "ai_edge_app" (
    echo ERROR: ai_edge_app folder not found!
    pause
    exit /b 1
)

cd ai_edge_app

echo [1/3] Installing critical dependencies...
pip install qrcode pillow onnxruntime==1.14.0 numpy opencv-python --quiet --disable-pip-version-check
if errorlevel 1 (
    echo WARNING: Some packages failed to install
)

echo [2/3] Installing remaining dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo WARNING: Some packages failed to install
    echo Continuing anyway...
)

echo.
echo [3/3] Starting Edge AI App...
echo.
echo ^>^> Camera window will open
echo ^>^> Press 'q' in camera window to quit
echo.

python main.py --camera 0
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start Edge AI App
    echo Common issues:
    echo   - Camera not connected
    echo   - Model file missing
    echo   - Python dependencies not installed
    echo.
    pause
)
