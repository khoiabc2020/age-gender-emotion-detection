@echo off
chcp 65001 >nul
echo ============================================================
echo PUSH CODE LEN GITHUB
echo ============================================================
echo.
echo Repository: https://github.com/khoiabc2020/age-gender-emotion-detection
echo Username: khoiabc2020
echo.

cd /d "%~dp0\.."

REM Setup remote
git remote remove origin 2>nul
git remote add origin https://github.com/khoiabc2020/age-gender-emotion-detection.git

REM Setup branch
git branch -M main 2>nul

REM Add và commit
git add .
git commit -m "Auto commit: %date% %time%" 2>nul

echo.
echo ============================================================
echo DANG PUSH LEN GITHUB...
echo ============================================================
echo.
echo LUU Y: GitHub KHONG chap nhan password!
echo Ban CAN Personal Access Token
echo.
echo Cach lay token:
echo 1. Mo: https://github.com/settings/tokens
echo 2. Generate new token ^> Generate new token (classic)
echo 3. Dat ten: "My Computer"
echo 4. Chon quyen: repo (full control)
echo 5. Generate va COPY TOKEN
echo.
echo Khi duoc hoi:
echo    Username: khoiabc2020
echo    Password: PASTE TOKEN VAO (khong phai password)
echo.
pause

git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Push that bai!
    echo Can Personal Access Token: https://github.com/settings/tokens
) else (
    echo.
    echo [SUCCESS] Da push thanh cong!
    echo Xem: https://github.com/khoiabc2020/age-gender-emotion-detection
)

pause

