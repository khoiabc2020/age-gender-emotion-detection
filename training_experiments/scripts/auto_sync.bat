@echo off
chcp 65001 >nul
echo ============================================================
echo 🔄 Tự động Sync Code lên GitHub
echo ============================================================
echo.

cd /d "%~dp0\.."

echo 📝 Đang kiểm tra thay đổi và sync lên GitHub...
python scripts\auto_git_push.py

echo.
pause

