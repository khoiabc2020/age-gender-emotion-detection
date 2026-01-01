@echo off
chcp 65001 >nul
cls
echo ======================================================
echo   CHECKING DEPENDENCIES
echo ======================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Backend API...
cd backend_api
python -c "import fastapi; print('  FastAPI: OK')" 2>nul || echo   FastAPI: NOT INSTALLED
python -c "import uvicorn; print('  Uvicorn: OK')" 2>nul || echo   Uvicorn: NOT INSTALLED
python -c "import sqlalchemy; print('  SQLAlchemy: OK')" 2>nul || echo   SQLAlchemy: NOT INSTALLED
cd ..

echo.
echo [2/3] Checking Dashboard...
cd dashboard
if exist "node_modules\" (
    echo   Node modules: OK
) else (
    echo   Node modules: NOT INSTALLED
)
cd ..

echo.
echo [3/3] Checking Edge AI App...
cd ai_edge_app
python -c "import cv2; print('  OpenCV: OK')" 2>nul || echo   OpenCV: NOT INSTALLED
python -c "import onnxruntime; print('  ONNX Runtime: OK')" 2>nul || echo   ONNX Runtime: NOT INSTALLED
python -c "import numpy; print('  NumPy: OK')" 2>nul || echo   NumPy: NOT INSTALLED
cd ..

echo.
echo ======================================================
echo   CHECK COMPLETE
echo ======================================================
echo.
echo If any dependencies are missing, run:
echo   INSTALL_DEPENDENCIES.bat
echo.
pause
