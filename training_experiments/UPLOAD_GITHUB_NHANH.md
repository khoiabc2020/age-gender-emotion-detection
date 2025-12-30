# ⚡ Upload Code lên GitHub - Hướng dẫn Nhanh

## 🚀 Cách Nhanh Nhất (3 bước)

### Bước 1: Tạo Repository trên GitHub
1. Truy cập: https://github.com/new
2. Đặt tên repo (ví dụ: `age-gender-emotion-detection`)
3. Click **Create repository**
4. **Copy URL** repo (ví dụ: `https://github.com/your-username/age-gender-emotion-detection.git`)

### Bước 2: Chạy Script Setup
```bash
# Windows
scripts\first_push.bat
```

Script sẽ hỏi:
- GitHub Username
- Repository Name

Sau đó tự động:
- ✅ Thêm remote
- ✅ Add files
- ✅ Commit
- ✅ Push lên GitHub

### Bước 3: Đăng nhập GitHub
Khi push, nhập:
- **Username**: GitHub username của bạn
- **Password**: Dùng **Personal Access Token** (không phải password thật)

**Lấy token**:
1. Truy cập: https://github.com/settings/tokens
2. Generate new token → Generate new token (classic)
3. Chọn quyền: **repo**
4. Generate và **copy token**

## 📝 Sau Khi Sửa Code

### Cách 1: Script Tự Động (Khuyến nghị)
```bash
scripts\auto_sync.bat
```

### Cách 2: Thủ Công
```bash
git add .
git commit -m "Mô tả thay đổi"
git push
```

## 🔄 Tự Động Sync (Real-time)

Chạy script này để tự động commit và push khi có thay đổi:
```bash
scripts\watch_sync.bat
```

## ❓ Gặp lỗi?

### Lỗi: "Git chưa được cài đặt"
- Tải Git: https://git-scm.com/downloads
- Cài đặt và chạy lại

### Lỗi: "authentication failed"
- Dùng Personal Access Token thay vì password
- Lấy token tại: https://github.com/settings/tokens

### Lỗi: "repository not found"
- Kiểm tra lại username và repo name
- Đảm bảo đã tạo repo trên GitHub

## 📚 Xem hướng dẫn chi tiết

Xem file: `HUONG_DAN_GITHUB.md`

---

**Tóm tắt**: 
1. Tạo repo trên GitHub
2. Chạy `scripts\first_push.bat`
3. Đăng nhập bằng token
4. Xong! 🎉

