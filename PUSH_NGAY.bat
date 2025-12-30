@echo off
chcp 65001 >nul
echo ============================================================
echo PUSH CODE LEN GITHUB - BAN CAN TOKEN
echo ============================================================
echo.
echo Repository: https://github.com/khoiabc2020/age-gender-emotion-detection
echo Username: khoiabc2020
echo.
echo ============================================================
echo QUAN TRONG: GitHub KHONG chap nhan password!
echo Ban CAN Personal Access Token
echo ============================================================
echo.
echo Cach lay token (2 phut):
echo 1. Mo: https://github.com/settings/tokens
echo 2. Click "Generate new token" ^> "Generate new token (classic)"
echo 3. Dat ten: "My Computer"
echo 4. Chon quyen: repo (full control)
echo 5. Click "Generate token"
echo 6. COPY TOKEN (chi hien 1 lan!)
echo.
echo Sau khi co token, chay lai script nay
echo Khi hoi password, PASTE TOKEN vao
echo.
pause

cd /d "%~dp0"

REM Setup branch
git branch -M main 2>nul

echo.
echo Dang push len GitHub...
echo.
echo Khi duoc hoi:
echo    Username: khoiabc2020
echo    Password: PASTE TOKEN VAO (khong phai password)
echo.

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
    echo 3. Chua tao repository tren GitHub
    echo.
    echo Giai phap:
    echo 1. Tao repository: https://github.com/new
    echo    Ten: age-gender-emotion-detection
    echo 2. Tao token: https://github.com/settings/tokens
    echo 3. Chay lai script nay
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
)

pause

