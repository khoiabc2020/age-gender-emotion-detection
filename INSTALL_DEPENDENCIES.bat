@echo off
chcp 65001 >nul
cls
echo ======================================================
echo   INSTALLING ALL DEPENDENCIES
echo ======================================================
echo.
echo This will install dependencies for:
echo   - Backend API (Python)
echo   - Dashboard (Node.js)
echo   - Edge AI App (Python)
echo.
echo This may take 5-10 minutes...
echo.
pause

cd /d "%~dp0"

echo.
echo ======================================================
echo [1/3] Backend API Dependencies
echo ======================================================
cd backend_api
if not exist ".env" (
    echo Creating .env file...
    (
        echo # Backend Configuration
        echo SECRET_KEY=your-secret-key-change-in-production-min-32-chars
        echo ALGORITHM=HS256
        echo ACCESS_TOKEN_EXPIRE_MINUTES=30
        echo CORS_ORIGINS=["http://localhost:3000"]
        echo DEBUG=True
    ) > .env
)
echo Installing Python packages...
pip install -r requirements.txt --upgrade --disable-pip-version-check -q 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Some packages failed to install. Retrying...
    pip install -r requirements.txt --upgrade --disable-pip-version-check
)
cd ..

echo.
echo ======================================================
echo [2/3] Dashboard Dependencies
echo ======================================================
cd dashboard
echo Installing Node packages (this may take a while)...
call npm install --silent --legacy-peer-deps 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Silent install failed. Retrying with verbose...
    call npm install --legacy-peer-deps
)
cd ..

echo.
echo ======================================================
echo [3/3] Edge AI App Dependencies
echo ======================================================
cd ai_edge_app
echo Installing core packages...
pip install qrcode pillow onnxruntime==1.16.0 numpy opencv-python requests paho-mqtt python-dotenv --upgrade --disable-pip-version-check -q 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Some packages failed. Retrying...
    pip install -r requirements.txt --upgrade --disable-pip-version-check
)
cd ..

echo.
echo ======================================================
echo   INSTALLATION COMPLETE!
echo ======================================================
echo.
echo You can now run:
echo   - START.bat (interactive menu)
echo   - run_app\run_all.bat (all services)
echo.
echo ======================================================
echo.
pause
