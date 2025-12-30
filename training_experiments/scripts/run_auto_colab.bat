@echo off
chcp 65001 >nul
echo ============================================================
echo 🚀 Tự động upload code lên Google Drive cho Colab
echo ============================================================
echo.

cd /d "%~dp0\.."

echo 📦 Bước 1: Tạo file zip từ code...
python scripts\upload_to_colab.py

echo.
echo ============================================================
echo ✅ Hoàn tất!
echo ============================================================
echo.
echo 📝 Các bước tiếp theo:
echo 1. Mở Google Colab: https://colab.research.google.com/
echo 2. Upload notebook: notebooks\train_on_colab_auto.ipynb
echo 3. Chọn GPU runtime: Runtime → Change runtime type → GPU
echo 4. Chạy tất cả cells (Runtime → Run all)
echo.
pause


