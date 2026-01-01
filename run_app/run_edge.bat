@echo off
chcp 65001 >nul
echo ======================================================
echo   EDGE AI APPLICATION
echo ======================================================
echo.

cd /d "%~dp0\.."
cd ai_edge_app

echo [1/3] Checking dependencies...
pip install qrcode pillow onnxruntime==1.14.0 --quiet

echo [2/3] Installing all dependencies...
pip install -r requirements.txt --quiet

echo.
echo [3/3] Starting Edge AI App...
echo Press 'q' to quit
echo.

python main.py --camera 0
