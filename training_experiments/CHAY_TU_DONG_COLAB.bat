@echo off
chcp 65001 >nul
echo ============================================================
echo 🚀 TỰ ĐỘNG UPLOAD CODE LÊN GOOGLE DRIVE CHO COLAB
echo ============================================================
echo.

cd /d "%~dp0"

echo 📦 Đang tạo file zip từ code...
python scripts\upload_to_colab.py

echo.
echo ============================================================
echo ✅ HOÀN TẤT!
echo ============================================================
echo.
echo 📝 CÁC BƯỚC TIẾP THEO:
echo.
echo 1. Mở Google Colab:
echo    https://colab.research.google.com/
echo.
echo 2. Upload notebook:
echo    notebooks\train_on_colab_auto.ipynb
echo.
echo 3. Chọn GPU runtime:
echo    Runtime → Change runtime type → GPU
echo.
echo 4. Chạy tất cả cells:
echo    Runtime → Run all (hoặc Ctrl+F9)
echo.
echo ============================================================
pause


