@echo off
chcp 65001 >nul
echo ======================================================
echo   EDGE AI APPLICATION
echo ======================================================
echo.

cd /d "%~dp0\.."
cd ai_edge_app

echo [1/2] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/2] Starting Edge AI App...
echo Press 'q' to quit
echo.

python main.py --camera 0
