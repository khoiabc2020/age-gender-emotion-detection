@echo off
chcp 65001 >nul
echo ============================================================
echo 🔄 Tiếp tục Push Code lên GitHub
echo ============================================================
echo.

cd /d "%~dp0\..\.."

REM Kiểm tra git status
echo 📊 Kiểm tra trạng thái...
git status

echo.
echo ============================================================
echo 💾 Đang commit...
echo ============================================================
echo.

REM Commit
git commit -m "Initial commit: Age Gender Emotion Detection"

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
echo    - Username: GitHub username của bạn
echo    - Password: Dùng Personal Access Token
echo    - Lấy token tại: https://github.com/settings/tokens
echo.
pause

git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Lỗi khi push!
    echo.
    echo 💡 Kiểm tra:
    echo    1. Đã tạo repository trên GitHub chưa?
    echo    2. Đã thêm remote chưa? (git remote -v)
    echo    3. Branch có phải 'main' không? (có thể là 'master')
    echo.
    echo 🔧 Thử lệnh:
    echo    git branch -M main
    echo    git push -u origin main
    echo.
) else (
    echo.
    echo ============================================================
    echo ✅ Đã push code lên GitHub thành công!
    echo ============================================================
    echo.
    REM Lấy URL repo
    for /f "tokens=2" %%i in ('git remote get-url origin') do set REPO_URL=%%i
    echo 🔗 Xem tại: %REPO_URL%
    echo.
)

pause

