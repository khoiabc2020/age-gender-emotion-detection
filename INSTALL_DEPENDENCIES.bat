@echo off
chcp 65001 >nul
echo ======================================================
echo   INSTALLING DEPENDENCIES
echo ======================================================
echo.

cd /d "%~dp0\.."

echo [1/3] Installing Backend dependencies...
cd backend_api
pip install -r requirements.txt --upgrade
cd ..

echo.
echo [2/3] Installing Dashboard dependencies...
cd dashboard
call npm install --legacy-peer-deps
cd ..

echo.
echo [3/3] Installing Edge AI dependencies...
cd ai_edge_app
pip install -r requirements.txt --upgrade
cd ..

echo.
echo ======================================================
echo   INSTALLATION COMPLETE
echo ======================================================
echo.
echo Now you can run:
echo   - START.bat (interactive menu)
echo   - run_app\run_all.bat (all services)
echo.
pause
