@echo off
REM Auto Training Optimizer - Chạy training 10 lần với tối ưu tự động
echo ========================================
echo AUTO TRAINING OPTIMIZER
echo ========================================
echo.
echo This will run training 10 times with different configurations
echo Each run will be optimized based on previous results
echo.
echo Estimated time: ~10-15 hours (depending on hardware)
echo.
pause

cd training_experiments

REM Activate virtual environment if exists
if exist venv_gpu\Scripts\activate.bat (
    call venv_gpu\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARNING] Virtual environment not found, using system Python
)

REM Check if data is preprocessed
echo.
echo [1/3] Checking preprocessed data...
if not exist "data\processed\utkface\train" (
    echo [ERROR] Preprocessed data not found!
    echo Please run: python src/data/preprocess.py
    pause
    exit /b 1
)
echo [OK] Data found

REM Run auto training optimizer
echo.
echo [2/3] Starting auto training optimizer...
echo This will run 10 training sessions with different configurations
echo.
python train_10x_automated.py

if errorlevel 1 (
    echo [ERROR] Auto training failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Training completed!
echo.
echo Results saved in: training_results/
echo   - all_results.json: All training results
echo   - best_config.json: Best configuration
echo   - run_*: Individual run directories
echo.
echo To view best model:
echo   Check training_results/run_<best_run_id>_*/best_model.pth
echo.
pause

