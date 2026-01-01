@echo off
chcp 65001 >nul
cls
echo ======================================================
echo   INSTALLING EDGE AI DEPENDENCIES
echo ======================================================
echo.

cd /d "%~dp0"
cd ai_edge_app

echo Checking Python version...
python -c "import sys; v=sys.version_info; print(f'Python {v.major}.{v.minor}.{v.micro}')"
echo.

echo ======================================================
echo [1/3] Installing Core Packages
echo ======================================================
pip install opencv-python numpy Pillow qrcode requests paho-mqtt python-dotenv --upgrade -q 2>nul
if %errorlevel% neq 0 (
    echo Retrying with verbose output...
    pip install opencv-python numpy Pillow qrcode requests paho-mqtt python-dotenv --upgrade
)
echo Core packages installed!
echo.

echo ======================================================
echo [2/3] Installing ONNX Runtime
echo ======================================================
echo Checking Python version for ONNX compatibility...
python -c "import sys; exit(0 if sys.version_info < (3, 13) else 1)" 2>nul
if %errorlevel% equ 0 (
    echo Python 3.12 or lower - Installing from PyPI...
    pip install onnxruntime>=1.15.0 --upgrade -q 2>nul
    if %errorlevel% neq 0 (
        pip install onnxruntime --upgrade
    )
) else (
    echo Python 3.13+ detected - Using alternative installation...
    echo.
    echo IMPORTANT: ONNX Runtime may not be available for Python 3.13+
    echo Trying latest version...
    pip install onnxruntime --upgrade 2>nul
    if %errorlevel% neq 0 (
        echo.
        echo ============================================================
        echo WARNING: ONNX Runtime installation failed!
        echo ============================================================
        echo.
        echo This is expected for Python 3.13+
        echo.
        echo SOLUTION OPTIONS:
        echo   1. Use Python 3.12 ^(Recommended^)
        echo      - Download: https://www.python.org/downloads/release/python-3120/
        echo.
        echo   2. Build from source ^(Advanced^)
        echo      - Follow: https://onnxruntime.ai/docs/build/
        echo.
        echo   3. Use Docker ^(Easiest^)
        echo      - Run: docker-compose up ai-edge-app
        echo.
        echo ============================================================
        pause
        goto :skip_onnx
    )
)
echo ONNX Runtime installed!
:skip_onnx
echo.

echo ======================================================
echo [3/3] Verifying Installation
echo ======================================================
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)" 2>nul || echo ✗ OpenCV: FAILED
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>nul || echo ✗ NumPy: FAILED
python -c "import onnxruntime; print('✓ ONNX Runtime:', onnxruntime.__version__)" 2>nul || echo ✗ ONNX Runtime: FAILED
python -c "import PIL; print('✓ Pillow:', PIL.__version__)" 2>nul || echo ✗ Pillow: FAILED
python -c "import qrcode; print('✓ QRCode: OK')" 2>nul || echo ✗ QRCode: FAILED
python -c "import paho.mqtt as mqtt; print('✓ MQTT: OK')" 2>nul || echo ✗ MQTT: FAILED
echo.

echo ======================================================
echo   INSTALLATION COMPLETE!
echo ======================================================
echo.
echo You can now run:
echo   run_app\run_edge.bat
echo.
pause
