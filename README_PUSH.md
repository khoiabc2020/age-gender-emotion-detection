# 🚀 Hướng dẫn Push Code lên GitHub - Tự Động

## ⚠️ QUAN TRỌNG: Cần Personal Access Token

GitHub **KHÔNG CÒN** chấp nhận password từ năm 2021.  
Bạn **PHẢI** dùng **Personal Access Token**.

## 📋 Các Bước (5 phút)

### Bước 1: Tạo Personal Access Token

1. Truy cập: **https://github.com/settings/tokens**
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Điền thông tin:
   - **Note**: "My Computer" (hoặc tên bất kỳ)
   - **Expiration**: Chọn "90 days" hoặc "No expiration"
   - **Select scopes**: Tích chọn **`repo`** (full control)
4. Click **"Generate token"**
5. **COPY TOKEN NGAY** (chỉ hiện 1 lần!)
   - Token có dạng: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Bước 2: Tạo Repository trên GitHub (nếu chưa có)

1. Truy cập: **https://github.com/new**
2. Repository name: `age-gender-emotion-detection`
3. Chọn **Public** hoặc **Private**
4. **KHÔNG** tích "Initialize with README"
5. Click **"Create repository"**

### Bước 3: Chạy Script Push

```bash
# Chạy script này
PUSH_NGAY.bat
```

Khi được hỏi:
- **Username**: `khoiabc2k4`
- **Password**: **PASTE TOKEN VÀO** (không phải password thật)

### Bước 4: Xác nhận

Sau khi push thành công, xem code tại:
**https://github.com/khoiabc2k4/age-gender-emotion-detection**

## 🔄 Sau Khi Push Thành Công

Để sync code sau này, chạy:
```bash
training_experiments\scripts\auto_sync.bat
```

Hoặc để tự động sync khi có thay đổi:
```bash
training_experiments\scripts\watch_sync.bat
```

## ❓ Gặp Lỗi?

### Lỗi: "authentication failed"
- ✅ Đảm bảo dùng **TOKEN** chứ không phải password
- ✅ Kiểm tra token còn hạn không
- ✅ Đảm bảo token có quyền **repo**

### Lỗi: "repository not found"
- ✅ Kiểm tra đã tạo repository trên GitHub chưa
- ✅ Kiểm tra username đúng: `khoiabc2k4`
- ✅ Kiểm tra tên repo đúng: `age-gender-emotion-detection`

### Lỗi: "remote origin already exists"
- ✅ Script đã tự động xử lý, không cần lo

## 📝 Tóm Tắt

1. ✅ Tạo token: https://github.com/settings/tokens
2. ✅ Tạo repo: https://github.com/new
3. ✅ Chạy: `PUSH_NGAY.bat`
4. ✅ Paste token khi hỏi password
5. ✅ Xong!

---

**Lưu ý**: Token là bí mật, đừng chia sẻ với ai!

