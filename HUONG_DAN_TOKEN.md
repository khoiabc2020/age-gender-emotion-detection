# 🔑 Hướng dẫn Lấy Personal Access Token

## ⚠️ QUAN TRỌNG

GitHub **KHÔNG CÒN** chấp nhận password thông thường từ năm 2021.  
Bạn **PHẢI** dùng **Personal Access Token** để push code.

## 🚀 Cách Lấy Token (3 phút)

### Bước 1: Truy cập Settings
1. Đăng nhập GitHub: https://github.com/login
2. Click avatar (góc phải trên) → **Settings**
3. Hoặc truy cập trực tiếp: https://github.com/settings/tokens

### Bước 2: Tạo Token
1. Scroll xuống phần **Developer settings** (bên trái)
2. Click **Personal access tokens** → **Tokens (classic)**
3. Click **Generate new token** → **Generate new token (classic)**

### Bước 3: Cấu hình Token
- **Note**: Đặt tên (ví dụ: "My Computer" hoặc "Windows PC")
- **Expiration**: Chọn thời hạn (90 days, hoặc No expiration)
- **Select scopes**: Tích chọn **repo** (full control of private repositories)
  - Điều này cho phép đọc/ghi repository

### Bước 4: Generate và Copy
1. Click **Generate token** (cuối trang)
2. **QUAN TRỌNG**: Token chỉ hiện **1 LẦN DUY NHẤT**
3. **COPY TOKEN NGAY** và lưu vào nơi an toàn
4. Token có dạng: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## 📝 Sử dụng Token

Khi push code và được hỏi:
- **Username**: `khoiabc2k4`
- **Password**: **PASTE TOKEN VÀO** (không phải password thật)

## 🔒 Bảo mật Token

- ✅ Lưu token ở nơi an toàn
- ✅ Không chia sẻ token với ai
- ✅ Nếu token bị lộ, xóa ngay và tạo token mới
- ✅ Có thể tạo nhiều token cho nhiều máy

## 🛠️ Tự động hóa (Tùy chọn)

Sau khi có token, có thể setup để không cần nhập lại:

### Cách 1: Git Credential Manager
```bash
git config --global credential.helper manager-core
```
Sau đó push 1 lần, token sẽ được lưu.

### Cách 2: GitHub CLI
```bash
# Cài đặt GitHub CLI
winget install GitHub.cli

# Đăng nhập
gh auth login
```

## ✅ Checklist

- [ ] Đã tạo Personal Access Token
- [ ] Đã copy và lưu token
- [ ] Đã chọn quyền **repo**
- [ ] Sẵn sàng paste token khi push

## 🚀 Sau khi có Token

Chạy script:
```bash
PUSH_TU_DONG.bat
```

Khi được hỏi password, **paste TOKEN vào** (không phải password thật).

---

**Lưu ý**: Token là bí mật, đừng commit token vào code!

