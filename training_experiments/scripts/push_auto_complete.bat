@echo off
chcp 65001 >nul
echo ============================================================
echo TU DONG PUSH CODE - HOAN TOAN TU DONG
echo ============================================================
echo.

cd /d "%~dp0\..\.."

REM Thông tin GitHub
set GITHUB_USER=khoiabc2k4
set GITHUB_REPO=age-gender-emotion-detection
set GITHUB_URL=https://github.com/%GITHUB_USER%/%GITHUB_REPO%.git

echo Repository: %GITHUB_URL%
echo Username: %GITHUB_USER%
echo.

REM Kiểm tra Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git chua duoc cai dat!
    pause
    exit /b 1
)

REM Setup git
if not exist ".git" (
    git init
)

REM Setup remote
git remote remove origin 2>nul
git remote add origin %GITHUB_URL%

REM Add và commit
git add .
git commit -m "Auto commit: %date% %time%" 2>nul

REM Setup branch
git branch -M main 2>nul

echo.
echo ============================================================
echo DANG PUSH...
echo ============================================================
echo.
echo CAN TOKEN: https://github.com/settings/tokens
echo.
echo Khi hoi password, paste TOKEN vao!
echo.
pause

git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Can Personal Access Token!
    echo Tao token: https://github.com/settings/tokens
) else (
    echo.
    echo [SUCCESS] Da push thanh cong!
    echo Xem: %GITHUB_URL%
)

pause

