@echo off
chcp 65001 >nul
echo ============================================================
echo TU DONG PUSH CODE LEN GITHUB
echo ============================================================
echo.
echo Repository: https://github.com/khoiabc2020/age-gender-emotion-detection
echo Username: khoiabc2020
echo.

cd /d "%~dp0"

REM Kiểm tra Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git chua duoc cai dat!
    echo Tai Git tai: https://git-scm.com/downloads
    pause
    exit /b 1
)

echo [OK] Git da duoc cai dat
echo.

REM Kiểm tra đã init chưa
if not exist ".git" (
    echo [1/6] Dang khoi tao git repository...
    git init
    echo [OK] Da khoi tao git repository
) else (
    echo [OK] Git repository da ton tai
)

echo.
echo [2/6] Dang setup remote...
git remote remove origin 2>nul
git remote add origin https://github.com/khoiabc2020/age-gender-emotion-detection.git
echo [OK] Da them remote

echo.
echo [3/6] Dang add files...
git add .
echo [OK] Da add files

echo.
echo [4/6] Dang commit...
git commit -m "Initial commit: Age Gender Emotion Detection Project" 2>nul
if errorlevel 1 (
    echo [WARN] Co the da commit roi hoac khong co thay doi
) else (
    echo [OK] Da commit
)

echo.
echo [5/6] Dang setup branch...
git branch -M main 2>nul
echo [OK] Da setup branch main

echo.
echo ============================================================
echo [6/6] DANG PUSH LEN GITHUB...
echo ============================================================
echo.
echo LUU Y QUAN TRONG:
echo GitHub KHONG con chap nhan password thong thuong!
echo Ban CAN dung Personal Access Token
echo.
echo Cach lay token:
echo 1. Truy cap: https://github.com/settings/tokens
echo 2. Click "Generate new token" ^> "Generate new token (classic)"
echo 3. Dat ten: "My Computer"
echo 4. Chon quyen: repo (full control)
echo 5. Click "Generate token"
echo 6. COPY TOKEN (chi hien 1 lan!)
echo.
echo Khi duoc hoi password, PASTE TOKEN vao (khong phai password)
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
    echo Co the do:
    echo 1. Chua co Personal Access Token
    echo 2. Token sai hoac het han
    echo 3. Repository chua duoc tao
    echo.
    echo Giai phap:
    echo 1. Tao token: https://github.com/settings/tokens
    echo 2. Chay lai script nay
    echo 3. Khi hoi password, paste TOKEN vao
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
    echo Xem code tai: https://github.com/khoiabc2k4/age-gender-emotion-detection
    echo.
    echo De sync code sau nay, chay:
    echo    training_experiments\scripts\auto_sync.bat
    echo.
)

pause

