@echo off
REM Standalone launcher for Edge AI App
REM This script ensures all dependencies are available before running

echo ============================================================
echo Smart Retail AI - Edge Application
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8-3.12
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "main_gui.py" (
    echo ERROR: main_gui.py not found!
    echo Please run this script from ai_edge_app directory
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models\multitask_model.onnx" (
    echo WARNING: Model file not found!
    echo Please ensure models/multitask_model.onnx exists
    pause
)

REM Check if configs exist
if not exist "configs\camera_config.json" (
    echo WARNING: Config file not found!
    echo Creating default config...
    mkdir configs 2>nul
    echo {} > configs\camera_config.json
)

REM Run the application
echo Starting Smart Retail AI...
echo.
python main_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start!
    echo Check logs/edge_app.log for details
    pause
)
