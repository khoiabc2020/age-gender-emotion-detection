# 🔄 Hướng dẫn Sync Code với GitHub cho Colab

## ❓ GitHub có tự động sync không?

**Trả lời**: Không, GitHub **KHÔNG** tự động sync theo thời gian thực. Bạn cần:
1. **Commit** thay đổi (lưu vào git)
2. **Push** lên GitHub (upload lên server)

## 🚀 Giải pháp: Tự động hóa

### Cách 1: Script tự động commit và push

Chạy script này sau khi sửa code:

```bash
# Windows
python scripts/auto_git_push.py

# Hoặc tạo file .bat để chạy nhanh
```

Script sẽ:
- ✅ Tự động phát hiện thay đổi
- ✅ Commit với timestamp
- ✅ Push lên GitHub

### Cách 2: Theo dõi thay đổi tự động (Real-time)

Chạy script này để tự động commit và push khi có thay đổi:

```bash
# Cài đặt watchdog (lần đầu)
pip install watchdog

# Chạy script theo dõi
python scripts/watch_and_push.py

# Hoặc với delay tùy chỉnh (30 giây)
python scripts/watch_and_push.py 60
```

Script sẽ:
- 👀 Theo dõi mọi thay đổi file
- ⏱️ Đợi 30 giây (tránh commit quá nhiều)
- 💾 Tự động commit và push

## 📋 Setup GitHub cho Colab

### Bước 1: Tạo GitHub Repository

1. Truy cập: https://github.com/new
2. Tạo repo mới (ví dụ: `age-gender-emotion-detection`)
3. Copy URL repo (ví dụ: `https://github.com/username/age-gender-emotion-detection.git`)

### Bước 2: Setup Git trên máy

```bash
# Nếu chưa có git repo
cd training_experiments
git init

# Thêm remote
git remote add origin https://github.com/your-username/your-repo.git

# Commit lần đầu
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Bước 3: Cập nhật Notebook Colab

Sửa trong notebook `train_on_colab_auto.ipynb`:

```python
# Cell "Download code từ Google Drive"
USE_GITHUB = True  # Bật GitHub
GITHUB_REPO_URL = "https://github.com/your-username/your-repo.git"  # URL repo của bạn
```

## 🔄 Workflow đề xuất

### Khi làm việc trên máy:

1. **Sửa code** như bình thường
2. **Chạy script tự động**:
   ```bash
   python scripts/auto_git_push.py
   ```
   Hoặc để script chạy tự động:
   ```bash
   python scripts/watch_and_push.py
   ```

### Khi train trên Colab:

1. **Mở notebook** trên Colab
2. **Chạy cell "Download code"** - Tự động pull code mới nhất từ GitHub
3. **Chạy training** như bình thường

## ⚙️ Tùy chỉnh

### Thay đổi delay (thời gian đợi trước khi commit)

```bash
# Đợi 60 giây trước khi commit
python scripts/watch_and_push.py 60
```

### Commit message tùy chỉnh

Sửa trong `auto_git_push.py`:
```python
commit_message = f"Your custom message: {timestamp}"
```

## 📝 Lưu ý

1. **GitHub không real-time**: Cần commit và push thủ công hoặc dùng script
2. **Colab pull mới nhất**: Mỗi lần chạy notebook, cell "Download code" sẽ pull code mới nhất
3. **Git credentials**: Lần đầu push cần đăng nhập GitHub
4. **.gitignore**: Đã tự động bỏ qua các file không cần thiết (data, checkpoints, logs)

## 🎯 Tóm tắt

| Hành động | Tự động? | Cách làm |
|-----------|----------|----------|
| Commit | ❌ | Chạy `auto_git_push.py` hoặc `watch_and_push.py` |
| Push | ❌ | Tự động khi chạy script |
| Pull trên Colab | ✅ | Tự động khi chạy cell "Download code" |
| Sync real-time | ⚠️ | Dùng `watch_and_push.py` (gần như real-time) |

## 🚀 Quick Start

1. **Setup GitHub repo** (lần đầu)
2. **Chạy script tự động**:
   ```bash
   python scripts/watch_and_push.py
   ```
3. **Làm việc bình thường** - Script sẽ tự động sync
4. **Train trên Colab** - Tự động pull code mới nhất

Chúc bạn làm việc hiệu quả! 🎉

