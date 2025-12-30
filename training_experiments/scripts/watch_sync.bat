@echo off
chcp 65001 >nul
echo ============================================================
echo 👀 Theo dõi thay đổi và tự động sync lên GitHub
echo ============================================================
echo.
echo 💡 Script sẽ tự động commit và push khi có thay đổi
echo 💡 Nhấn Ctrl+C để dừng
echo.

cd /d "%~dp0\.."

REM Kiểm tra watchdog
python -c "import watchdog" 2>nul
if errorlevel 1 (
    echo 📦 Đang cài đặt watchdog...
    pip install watchdog
)

echo.
echo 🚀 Bắt đầu theo dõi...
python scripts\watch_and_push.py

pause

