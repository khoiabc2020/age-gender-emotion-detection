@echo off
chcp 65001 >nul
echo ============================================================
echo 📤 Push Code lên GitHub Repository
echo ============================================================
echo.
echo Repository: https://github.com/khoile2k4/age-gender-emotion-detection
echo.

cd /d "%~dp0\..\.."

REM Kiểm tra Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git chưa được cài đặt!
    echo.
    echo 📥 Tải Git tại: https://git-scm.com/downloads
    pause
    exit /b 1
)

echo ✅ Git đã được cài đặt
echo.

REM Kiểm tra đã init chưa
if not exist ".git" (
    echo 🔧 Đang khởi tạo git repository...
    git init
    echo ✅ Đã khởi tạo git repository
)

echo.
echo ============================================================
echo 🔧 Đang setup remote...
echo ============================================================
echo.

REM Kiểm tra remote đã có chưa
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo 📍 Đang thêm remote...
    git remote add origin https://github.com/khoile2k4/age-gender-emotion-detection.git
    echo ✅ Đã thêm remote
) else (
    echo ⚠️  Remote đã tồn tại
    git remote set-url origin https://github.com/khoile2k4/age-gender-emotion-detection.git
    echo ✅ Đã cập nhật remote
)

echo.
echo 📦 Đang add files...
git add .

echo.
echo 💾 Đang commit...
git commit -m "Initial commit: Age Gender Emotion Detection Project"

if errorlevel 1 (
    echo.
    echo ⚠️  Có thể đã commit rồi hoặc không có thay đổi
    echo.
)

echo.
echo ============================================================
echo 📤 Đang push lên GitHub...
echo ============================================================
echo.
echo ⚠️  LƯU Ý: Lần đầu push sẽ yêu cầu đăng nhập GitHub
echo    - Username: khoile2k4
echo    - Password: Dùng Personal Access Token (KHÔNG phải password)
echo.
echo 📝 Lấy token:
echo    1. Truy cập: https://github.com/settings/tokens
echo    2. Generate new token → Generate new token (classic)
echo    3. Chọn quyền: repo (full control)
echo    4. Generate và copy token
echo.
pause

REM Thử push với branch main
git push -u origin main

if errorlevel 1 (
    echo.
    echo ⚠️  Thử với branch master...
    git branch -M main 2>nul
    git push -u origin main
    
    if errorlevel 1 (
        echo.
        echo ❌ Lỗi khi push!
        echo.
        echo 💡 Có thể do:
        echo    1. Chưa đăng nhập GitHub
        echo    2. Sai token hoặc password
        echo    3. Repository chưa được tạo đúng
        echo.
        echo 🔧 Thử lại:
        echo    git push -u origin main
        echo.
    ) else (
        echo.
        echo ============================================================
        echo ✅ Đã push code lên GitHub thành công!
        echo ============================================================
        echo.
        echo 🔗 Xem tại: https://github.com/khoile2k4/age-gender-emotion-detection
        echo.
    )
) else (
    echo.
    echo ============================================================
    echo ✅ Đã push code lên GitHub thành công!
    echo ============================================================
    echo.
    echo 🔗 Xem tại: https://github.com/khoile2k4/age-gender-emotion-detection
    echo.
)

pause

