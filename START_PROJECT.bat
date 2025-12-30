@echo off
REM ========================================
REM SMART RETAIL AI - START PROJECT
REM ========================================
echo.
echo ========================================
echo SMART RETAIL AI - COMPLETE SYSTEM
echo ========================================
echo.
echo This script will help you start the project
echo.
echo Options:
echo   1. Training Test (Quick test training pipeline)
echo   2. Backend API (FastAPI server)
echo   3. Frontend Dashboard (React app)
echo   4. All Services (Backend + Frontend)
echo   5. Exit
echo.
set /p choice="Select option (1-5): "

if "%choice%"=="1" goto TRAINING
if "%choice%"=="2" goto BACKEND
if "%choice%"=="3" goto FRONTEND
if "%choice%"=="4" goto ALL
if "%choice%"=="5" goto END
goto END

:TRAINING
echo.
echo Starting Training Test...
call run_training_test.bat
goto END

:BACKEND
echo.
echo Starting Backend API...
start "Backend API" cmd /k "cd /d %~dp0 && run_backend.bat"
echo.
echo Backend started in new window!
echo Access: http://localhost:8000/docs
goto END

:FRONTEND
echo.
echo Starting Frontend Dashboard...
start "Frontend Dashboard" cmd /k "cd /d %~dp0 && run_frontend.bat"
echo.
echo Frontend started in new window!
echo Access: http://localhost:3000
echo Login: admin / admin123
goto END

:ALL
echo.
echo Starting All Services...
start "Backend API" cmd /k "cd /d %~dp0 && run_backend.bat"
timeout /t 3 /nobreak >nul
start "Frontend Dashboard" cmd /k "cd /d %~dp0 && run_frontend.bat"
echo.
echo All services started!
echo   - Backend: http://localhost:8000
echo   - Frontend: http://localhost:3000
echo   - Login: admin / admin123
goto END

:END
echo.
pause

