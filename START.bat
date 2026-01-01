@echo off
chcp 65001 >nul
cls
echo.
echo ======================================================
echo   SMART RETAIL AI - QUICK LAUNCHER
echo ======================================================
echo.
echo Select an option:
echo.
echo   1. Run ALL Services (Backend + Frontend + Edge)
echo   2. Run Backend API only
echo   3. Run Dashboard only
echo   4. Run Edge AI App only
echo   5. Help / Documentation
echo   0. Exit
echo.
echo ======================================================
echo.

set /p choice="Enter your choice (0-5): "

if "%choice%"=="1" (
    echo.
    echo Starting ALL services...
    call run_app\run_all.bat
) else if "%choice%"=="2" (
    echo.
    echo Starting Backend API...
    call run_app\run_backend.bat
) else if "%choice%"=="3" (
    echo.
    echo Starting Dashboard...
    call run_app\run_frontend.bat
) else if "%choice%"=="4" (
    echo.
    echo Starting Edge AI App...
    call run_app\run_edge.bat
) else if "%choice%"=="5" (
    echo.
    echo Opening documentation...
    start README.md
    start QUICKSTART.md
    echo.
    pause
    call %0
) else if "%choice%"=="0" (
    echo.
    echo Goodbye!
    exit
) else (
    echo.
    echo Invalid choice! Please try again.
    timeout /t 2 >nul
    call %0
)
