@echo off
REM Frontend Dashboard - React
echo ========================================
echo STARTING FRONTEND DASHBOARD
echo ========================================
echo.

cd dashboard

REM Check if node_modules exists
if not exist node_modules (
    echo [INFO] Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed!
        pause
        exit /b 1
    )
) else (
    REM Check if vite is installed
    if not exist "node_modules\.bin\vite.cmd" (
        echo [WARNING] Vite not found, reinstalling dependencies...
        call npm install
        if errorlevel 1 (
            echo [ERROR] npm install failed!
            pause
            exit /b 1
        )
    )
)

REM Check if .env.local exists
if not exist .env.local (
    echo [INFO] Creating .env.local...
    (
        echo VITE_API_BASE_URL=http://localhost:8000
    ) > .env.local
    echo [OK] .env.local created
)

echo.
echo [INFO] Starting React development server...
echo [INFO] Dashboard will be available at: http://localhost:3000
echo [INFO] Default login: admin / admin123
echo.
echo Press Ctrl+C to stop
echo.

REM Use npx vite (more reliable, works even if vite not in node_modules)
call npx vite

