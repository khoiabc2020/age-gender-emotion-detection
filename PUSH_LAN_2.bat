@echo off
chcp 65001 >nul
echo ============================================================
echo PUSH CODE LEN GITHUB - TAI KHOAN MOI
echo ============================================================
echo.
echo Repository: https://github.com/khoiabc2020/age-gender-emotion-detection
echo Username: khoiabc2020
echo.

cd /d "%~dp0"

REM Setup remote
git remote remove origin 2>nul
git remote add origin https://github.com/khoiabc2020/age-gender-emotion-detection.git

REM Setup branch
git branch -M main 2>nul

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

echo.
echo Dang push...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] Push that bai!
    echo ============================================================
    echo.
    echo Nguyen nhan co the:
    echo 1. Chua co Personal Access Token
    echo 2. Token sai hoac het han
    echo 3. Repository chua duoc tao
    echo.
    echo Giai phap:
    echo 1. Tao repository: https://github.com/new
    echo    Ten: age-gender-emotion-detection
    echo 2. Tao token: https://github.com/settings/tokens
    echo 3. Chay lai script nay
    echo.
    echo Hoac thu push thu cong:
    echo    git push -u origin main
    echo.
) else (
    echo.
    echo ============================================================
    echo [SUCCESS] DA PUSH CODE LEN GITHUB THANH CONG!
    echo ============================================================
    echo.
    echo Xem code tai:
    echo https://github.com/khoiabc2020/age-gender-emotion-detection
    echo.
    echo De sync code sau nay:
    echo    training_experiments\scripts\auto_sync.bat
    echo.
)

pause

