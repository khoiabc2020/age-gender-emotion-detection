@echo off
chcp 65001 >nul
echo ============================================================
echo 🔧 Setup GitHub Repository (Lần đầu)
echo ============================================================
echo.

cd /d "%~dp0\..\.."

REM Kiểm tra Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git chưa được cài đặt!
    echo.
    echo 📥 Tải Git tại: https://git-scm.com/downloads
    echo.
    pause
    exit /b 1
)

echo ✅ Git đã được cài đặt
echo.

REM Kiểm tra đã init chưa
if exist ".git" (
    echo ✅ Đã là git repository
) else (
    echo 🔧 Đang khởi tạo git repository...
    git init
    echo ✅ Đã khởi tạo git repository
)

echo.
echo ============================================================
echo 📝 Các bước tiếp theo:
echo ============================================================
echo.
echo 1. Tạo repository trên GitHub:
echo    https://github.com/new
echo.
echo 2. Copy URL repository (ví dụ:)
echo    https://github.com/your-username/your-repo.git
echo.
echo 3. Chạy lệnh sau (thay YOUR_USERNAME và YOUR_REPO):
echo.
echo    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
echo    git add .
echo    git commit -m "Initial commit"
echo    git push -u origin main
echo.
echo ============================================================
echo.
echo 💡 Hoặc chạy script tự động:
echo    scripts\auto_sync.bat
echo.
pause

