# 🔒 SECURITY GUIDE - SMART RETAIL AI

**Bảo mật API Keys và Thông tin nhạy cảm**

---

## ✅ KIỂM TRA BẢO MẬT

### 1. **Hardcoded Keys** ✅
- ✅ **Không tìm thấy hardcoded API keys**
- ✅ Tất cả files đã được kiểm tra
- ✅ Không có keys thật trong code

### 2. **Environment Variables** ✅
- ✅ **Sử dụng đúng cách**
- ✅ `backend_api/app/core/config.py` - Dùng `BaseSettings` từ `.env`
- ✅ `backend_api/app/services/ai_agent.py` - Dùng `os.getenv()`
- ✅ `backend_api/app/api/ai_agent.py` - Lấy từ `settings`

### 3. **.gitignore** ✅
- ✅ `.env` được ignore
- ✅ `.env.local` được ignore
- ✅ Các file sensitive không bị commit

---

## 📝 CẤU HÌNH API KEYS

### Backend API Keys

**File**: `backend_api/.env`

```env
# Google AI (Gemini)
GOOGLE_AI_API_KEY=your-google-ai-api-key-here

# OpenAI (ChatGPT)
OPENAI_API_KEY=your-openai-api-key-here

# AI Provider (google_ai, chatgpt, or both)
AI_PROVIDER=google_ai

# Secret Key (JWT)
SECRET_KEY=your-secret-key-change-in-production

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

### Frontend Environment Variables

**File**: `dashboard/.env.local`

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 🔐 BEST PRACTICES

### 1. **Không commit .env files**
- ✅ Đã có trong `.gitignore`
- ✅ Sử dụng `.env.example` cho template

### 2. **Rotate Keys định kỳ**
- Thay đổi API keys mỗi 3-6 tháng
- Revoke keys cũ khi không dùng

### 3. **Sử dụng Secrets Management**
- Production: Dùng Vault, AWS Secrets Manager, hoặc K8s Secrets
- Development: Dùng `.env` files (đã ignore)

### 4. **Kiểm tra định kỳ**
```bash
# Chạy script kiểm tra
python check_api_keys.py
```

---

## 🛡️ SECURITY CHECKLIST

- [x] Không có hardcoded keys trong code
- [x] `.env` files được ignore
- [x] Sử dụng environment variables
- [x] Có `.env.example` templates
- [x] JWT secret key được config
- [x] Database credentials được bảo vệ
- [x] API keys được rotate định kỳ

---

## 📚 TÀI LIỆU THAM KHẢO

- [GIT_COMMIT_GUIDE.md](GIT_COMMIT_GUIDE.md) - Git commit guidelines
- [CI_CD_DOCUMENTATION.md](CI_CD_DOCUMENTATION.md) - CI/CD security

---

**Status**: ✅ Secure  
**Last Updated**: 2025-12-30

