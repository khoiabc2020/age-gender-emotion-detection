@echo off
chcp 65001 >nul
echo ============================================================
echo Push Code len GitHub Repository
echo ============================================================
echo.
echo Repository: https://github.com/khoile2k4/age-gender-emotion-detection
echo.

cd /d "%~dp0"

REM Kiểm tra Git
git --version >nul 2>&1
if errorlevel 1 (
    echo Git chua duoc cai dat!
    pause
    exit /b 1
)

REM Kiểm tra đã init chưa
if not exist ".git" (
    echo Dang khoi tao git repository...
    git init
)

REM Setup remote
git remote remove origin 2>nul
git remote add origin https://github.com/khoile2k4/age-gender-emotion-detection.git

echo.
echo Dang add files...
git add .

echo.
echo Dang commit...
git commit -m "Initial commit: Age Gender Emotion Detection Project"

echo.
echo ============================================================
echo Dang push len GitHub...
echo ============================================================
echo.
echo LUU Y: Lan dau push se yeu cau dang nhap GitHub
echo    - Username: khoile2k4
echo    - Password: Dung Personal Access Token
echo.
echo Lay token: https://github.com/settings/tokens
echo.
pause

git branch -M main 2>nul
git push -u origin main

if errorlevel 1 (
    echo.
    echo Co the can dang nhap GitHub hoac token
    echo Thu lai: git push -u origin main
) else (
    echo.
    echo ============================================================
    echo Da push code len GitHub thanh cong!
    echo ============================================================
    echo.
    echo Xem tai: https://github.com/khoile2k4/age-gender-emotion-detection
    echo.
)

pause

