@echo off
chcp 65001 >nul
cls

:MENU
cls
echo.
echo ======================================================
echo        SMART RETAIL AI - QUICK LAUNCHER
echo ======================================================
echo.
echo   Select an option:
echo.
echo   1. Start ALL Services (Backend + Dashboard + Edge)
echo   2. Start Backend API only
echo   3. Start Dashboard only
echo   4. Start Edge AI App only
echo   5. Install All Dependencies
echo   6. Help / Documentation
echo   0. Exit
echo.
echo ======================================================
echo.

set /p choice="Enter your choice (0-6): "

if "%choice%"=="1" goto RUN_ALL
if "%choice%"=="2" goto RUN_BACKEND
if "%choice%"=="3" goto RUN_FRONTEND
if "%choice%"=="4" goto RUN_EDGE
if "%choice%"=="5" goto INSTALL
if "%choice%"=="6" goto HELP
if "%choice%"=="0" goto EXIT
goto INVALID

:RUN_ALL
echo.
echo Starting ALL services...
call run_app\run_all.bat
goto END

:RUN_BACKEND
echo.
echo Starting Backend API...
call run_app\run_backend.bat
goto END

:RUN_FRONTEND
echo.
echo Starting Dashboard...
call run_app\run_frontend.bat
goto END

:RUN_EDGE
echo.
echo Starting Edge AI App...
call run_app\run_edge.bat
goto END

:INSTALL
echo.
echo Installing all dependencies...
call INSTALL_DEPENDENCIES.bat
goto MENU

:HELP
cls
echo.
echo ======================================================
echo        DOCUMENTATION
echo ======================================================
echo.
echo Main Files:
echo   README.md           - Project overview
echo   QUICKSTART.md       - Quick start guide
echo   FINAL_STATUS.md     - Project status
echo.
echo Access URLs:
echo   Backend:   http://localhost:8000/docs
echo   Dashboard: http://localhost:3000
echo.
echo Login:
echo   Username: admin
echo   Password: admin123
echo.
echo ======================================================
echo.
pause
goto MENU

:INVALID
echo.
echo Invalid choice! Please try again.
timeout /t 2 >nul
goto MENU

:EXIT
echo.
echo Goodbye!
timeout /t 1 >nul
exit

:END
echo.
pause
goto MENU
