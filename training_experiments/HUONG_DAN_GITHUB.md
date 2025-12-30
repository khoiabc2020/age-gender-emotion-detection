# 📤 Hướng dẫn Upload Code lên GitHub

## 🚀 Cách 1: Sử dụng Script Tự Động (Khuyến nghị)

### Bước 1: Chạy script tự động
```bash
# Windows
scripts\auto_sync.bat

# Hoặc Python
python scripts/auto_git_push.py
```

Script sẽ tự động:
- ✅ Kiểm tra thay đổi
- ✅ Commit code
- ✅ Push lên GitHub

## 📋 Cách 2: Làm Thủ Công (Lần đầu setup)

### Bước 1: Tạo Repository trên GitHub

1. Truy cập: https://github.com/new
2. Điền thông tin:
   - **Repository name**: `age-gender-emotion-detection` (hoặc tên bạn muốn)
   - **Description**: "Age, Gender, and Emotion Detection using Deep Learning"
   - **Public** hoặc **Private** (tùy chọn)
   - **KHÔNG** tích "Initialize with README" (vì bạn đã có code)
3. Click **Create repository**

### Bước 2: Setup Git trên máy tính

#### Kiểm tra Git đã cài chưa:
```bash
git --version
```

Nếu chưa có, tải tại: https://git-scm.com/downloads

#### Khởi tạo Git Repository:

```bash
# Di chuyển vào thư mục project
cd "D:\AI vietnam\Code\nhan dien do tuoi"

# Khởi tạo git (nếu chưa có)
git init

# Kiểm tra trạng thái
git status
```

### Bước 3: Tạo .gitignore (Bỏ qua file không cần thiết)

File `.gitignore` đã được tạo tự động bởi script, hoặc tạo thủ công:

```bash
# Tạo file .gitignore
notepad .gitignore
```

Nội dung `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
venv_gpu/
env/
ENV/

# Data
data/
*.zip
*.pth
*.onnx
*.h5
*.ckpt

# Logs
logs/
*.log

# Checkpoints
checkpoints/
results/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Temporary
*.tmp
*.temp
```

### Bước 4: Thêm Remote và Push

```bash
# Thêm remote (thay YOUR_USERNAME và YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Kiểm tra remote
git remote -v

# Add tất cả file
git add .

# Commit lần đầu
git commit -m "Initial commit: Age Gender Emotion Detection"

# Push lên GitHub
git push -u origin main
```

**Lưu ý**: 
- Nếu branch là `master` thay vì `main`, dùng: `git push -u origin master`
- Lần đầu push sẽ yêu cầu đăng nhập GitHub

## 🔐 Xác thực GitHub

### Cách 1: Personal Access Token (Khuyến nghị)

1. Truy cập: https://github.com/settings/tokens
2. Click **Generate new token** → **Generate new token (classic)**
3. Đặt tên token (ví dụ: "My Computer")
4. Chọn quyền: **repo** (full control)
5. Click **Generate token**
6. **Copy token** (chỉ hiện 1 lần!)

Khi push, dùng token thay vì password:
- Username: GitHub username của bạn
- Password: Paste token vừa copy

### Cách 2: GitHub CLI

```bash
# Cài đặt GitHub CLI
# Windows: winget install GitHub.cli
# Hoặc tải: https://cli.github.com/

# Đăng nhập
gh auth login

# Sau đó push bình thường
git push
```

### Cách 3: SSH Key (Nâng cao)

1. Tạo SSH key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Copy public key:
```bash
cat ~/.ssh/id_ed25519.pub
```

3. Thêm vào GitHub:
   - Settings → SSH and GPG keys → New SSH key
   - Paste public key

4. Đổi remote sang SSH:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
```

## 🔄 Push Code Sau Khi Sửa

### Cách 1: Script Tự Động (Khuyến nghị)

```bash
# Chạy script
scripts\auto_sync.bat
```

### Cách 2: Lệnh Git Thủ Công

```bash
# Kiểm tra thay đổi
git status

# Add file đã sửa
git add .

# Commit
git commit -m "Mô tả thay đổi"

# Push
git push
```

## 📝 Commit Message Tốt

Viết commit message rõ ràng:

```bash
# Tốt
git commit -m "Add Colab training notebook"
git commit -m "Fix dataset loading error"
git commit -m "Update model architecture"

# Không tốt
git commit -m "update"
git commit -m "fix"
git commit -m "changes"
```

## 🛠️ Troubleshooting

### Lỗi: "remote origin already exists"
```bash
# Xóa remote cũ
git remote remove origin

# Thêm lại
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Lỗi: "failed to push some refs"
```bash
# Pull code mới nhất trước
git pull origin main --rebase

# Sau đó push lại
git push
```

### Lỗi: "authentication failed"
- Kiểm tra lại Personal Access Token
- Hoặc dùng GitHub CLI: `gh auth login`

### Lỗi: "branch 'main' does not exist"
```bash
# Tạo branch main
git branch -M main

# Push
git push -u origin main
```

## 🎯 Quick Start (Tóm tắt)

### Lần đầu:
```bash
# 1. Tạo repo trên GitHub
# 2. Chạy các lệnh:
cd "D:\AI vietnam\Code\nhan dien do tuoi"
git init
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Sau khi sửa code:
```bash
# Cách 1: Script tự động
scripts\auto_sync.bat

# Cách 2: Thủ công
git add .
git commit -m "Mô tả thay đổi"
git push
```

## 📚 Tài liệu tham khảo

- Git Documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com/
- Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf

## ✅ Checklist

- [ ] Đã tạo GitHub account
- [ ] Đã tạo repository trên GitHub
- [ ] Đã cài Git trên máy
- [ ] Đã khởi tạo git repo (`git init`)
- [ ] Đã tạo `.gitignore`
- [ ] Đã thêm remote (`git remote add origin`)
- [ ] Đã push code lần đầu (`git push -u origin main`)
- [ ] Đã setup Personal Access Token
- [ ] Đã test push thành công

Chúc bạn thành công! 🎉

