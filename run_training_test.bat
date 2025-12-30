@echo off
REM Training Test Script - Chạy test training pipeline
echo ========================================
echo TRAINING TEST - Smart Retail AI
echo ========================================
echo.

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
echo [1/4] Checking preprocessed data...
python -c "from pathlib import Path; import sys; d = Path('data/processed/utkface'); print('✅ Data found' if d.exists() else '❌ Data not found - run preprocessing first'); sys.exit(0 if d.exists() else 1)"
if errorlevel 1 (
    echo.
    echo [INFO] Running preprocessing...
    python src/data/preprocess.py
    if errorlevel 1 (
        echo [ERROR] Preprocessing failed!
        pause
        exit /b 1
    )
)

REM Run test pipeline
echo.
echo [2/4] Running test pipeline...
python scripts/test_pipeline.py
if errorlevel 1 (
    echo [ERROR] Test pipeline failed!
    pause
    exit /b 1
)

REM Quick training test (1 epoch)
echo.
echo [3/4] Running quick training test (1 epoch)...
python train_optimized.py ^
    --data_dir data/processed/utkface ^
    --batch_size 16 ^
    --epochs 1 ^
    --lr 1e-3 ^
    --num_workers 2 ^
    --save_dir ./checkpoints_test ^
    --early_stop_patience 999
if errorlevel 1 (
    echo [ERROR] Training test failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Training test completed successfully!
echo.
echo Next steps:
echo   1. Check checkpoints_test/ for saved model
echo   2. Run full training: python train_optimized.py --epochs 60
echo   3. Monitor with: tensorboard --logdir checkpoints_test/logs
echo.
pause
