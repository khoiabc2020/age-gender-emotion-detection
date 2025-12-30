@echo off
chcp 65001 >nul
echo ============================================================
echo 📤 Push Code lên GitHub (Lần đầu)
echo ============================================================
echo.

cd /d "%~dp0\..\.."

REM Kiểm tra Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git chưa được cài đặt!
    pause
    exit /b 1
)

REM Kiểm tra đã init chưa
if not exist ".git" (
    echo ❌ Chưa khởi tạo git repository!
    echo.
    echo Chạy: scripts\setup_github.bat
    pause
    exit /b 1
)

echo 📝 Nhập thông tin GitHub repository:
echo.
set /p GITHUB_USERNAME="GitHub Username: "
set /p GITHUB_REPO="Repository Name: "

if "%GITHUB_USERNAME%"=="" (
    echo ❌ Username không được để trống!
    pause
    exit /b 1
)

if "%GITHUB_REPO%"=="" (
    echo ❌ Repository name không được để trống!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🔧 Đang setup...
echo ============================================================
echo.

REM Kiểm tra remote đã có chưa
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo 📍 Đang thêm remote...
    git remote add origin https://github.com/%GITHUB_USERNAME%/%GITHUB_REPO%.git
    echo ✅ Đã thêm remote
) else (
    echo ⚠️  Remote đã tồn tại
    set /p OVERWRITE="Bạn có muốn ghi đè? (y/n): "
    if /i "%OVERWRITE%"=="y" (
        git remote set-url origin https://github.com/%GITHUB_USERNAME%/%GITHUB_REPO%.git
        echo ✅ Đã cập nhật remote
    )
)

echo.
echo 📦 Đang add files...
git add .

echo.
echo 💾 Đang commit...
git commit -m "Initial commit: Age Gender Emotion Detection"

echo.
echo 📤 Đang push lên GitHub...
echo.
echo ⚠️  LƯU Ý: Lần đầu push sẽ yêu cầu đăng nhập GitHub
echo    - Username: %GITHUB_USERNAME%
echo    - Password: Dùng Personal Access Token (không phải password)
echo    - Lấy token tại: https://github.com/settings/tokens
echo.
pause

git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Lỗi khi push!
    echo.
    echo 💡 Có thể do:
    echo    1. Chưa tạo repository trên GitHub
    echo    2. Sai username/repo name
    echo    3. Chưa đăng nhập GitHub
    echo    4. Branch không phải 'main' (có thể là 'master')
    echo.
) else (
    echo.
    echo ============================================================
    echo ✅ Đã push code lên GitHub thành công!
    echo ============================================================
    echo.
    echo 🔗 Xem tại: https://github.com/%GITHUB_USERNAME%/%GITHUB_REPO%
    echo.
)

pause

