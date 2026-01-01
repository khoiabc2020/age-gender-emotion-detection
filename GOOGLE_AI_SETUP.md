# 🤖 Google Gemini API Setup Guide

## 🎯 **Mục đích**

Cấu hình Google Gemini AI để sử dụng cho Chat Agent trong hệ thống Smart Retail Analytics.

---

## 📋 **Bước 1: Lấy Google API Key**

### **1.1. Truy cập Google AI Studio**
```
https://makersuite.google.com/app/apikey
```

### **1.2. Tạo API Key**
1. Đăng nhập bằng Google Account
2. Click **"Create API Key"**
3. Chọn Google Cloud Project (hoặc tạo mới)
4. Copy API Key

**Ví dụ API Key:**
```
AIzaSyABC123def456GHI789jkl012MNO345pqr
```

---

## 🔧 **Bước 2: Cấu hình Backend**

### **2.1. Tạo file `.env`**
```bash
cd backend_api
copy .env.example .env  # Windows
# hoặc
cp .env.example .env    # Linux/Mac
```

### **2.2. Thêm API Key vào `.env`**
```bash
# Mở file backend_api/.env và thêm:

# AI Agent Configuration
GOOGLE_AI_API_KEY=AIzaSyABC123def456GHI789jkl012MNO345pqr
AI_PROVIDER=google_ai
```

### **2.3. File `.env` hoàn chỉnh**
```env
# Backend Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-secret-key-change-in-production-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/retail_analytics

# AI Agent Configuration (Google Gemini)
GOOGLE_AI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
AI_PROVIDER=google_ai
```

---

## 🚀 **Bước 3: Khởi động Backend**

```bash
START.bat → [3] Run Backend
```

Hoặc:
```bash
cd backend_api
python -m app.main
```

---

## 🧪 **Bước 4: Test AI Agent**

### **4.1. Truy cập API Docs**
```
http://localhost:8000/docs
```

### **4.2. Test endpoint `/api/v1/ai/analyze`**
```json
POST /api/v1/ai/analyze
{
  "time_range_hours": 24
}
```

**Response:**
```json
{
  "analysis": "Based on the analytics data...",
  "insights": [...],
  "recommendations": [...]
}
```

### **4.3. Test endpoint `/api/v1/ai/chat`**
```json
POST /api/v1/ai/chat
{
  "message": "What are the top selling products?",
  "time_range_hours": 24
}
```

**Response:**
```json
{
  "response": "Based on the data, the top selling products are...",
  "timestamp": "2025-01-02T..."
}
```

---

## ✅ **Xác nhận cấu hình**

### **Check API Status:**
```
GET http://localhost:8000/api/v1/ai/status
```

**Response:**
```json
{
  "google_ai_configured": true,
  "openai_configured": false,
  "provider": "google_ai"
}
```

---

## 💡 **Free Tier Limits**

**Google Gemini API (Free):**
- ✅ 60 requests/minute
- ✅ 1500 requests/day
- ✅ No credit card required

**Đủ cho:**
- Development & Testing
- Small production apps
- Personal projects

---

## 🔒 **Bảo mật API Key**

### **❌ KHÔNG làm:**
```bash
# Đừng commit .env vào git
git add .env  # ❌ WRONG!
```

### **✅ Nên làm:**
```bash
# .env đã được thêm vào .gitignore
# Chỉ commit .env.example

# Kiểm tra:
cat .gitignore | grep .env
# Output: .env
```

### **Production:**
- Dùng Environment Variables
- Hoặc Secret Management (AWS Secrets Manager, Azure Key Vault)

---

## 📊 **Tính năng AI Agent**

### **1. Analytics Analysis**
```python
# Phân tích dữ liệu analytics tự động
POST /api/v1/ai/analyze
```
- Phân tích xu hướng khách hàng
- Insights về demographics
- Recommendations cho quảng cáo

### **2. Chat Interface**
```python
# Chat với AI về dữ liệu
POST /api/v1/ai/chat
```
- Hỏi đáp về analytics
- Query dữ liệu tự nhiên
- Explanations & insights

### **3. Ad Optimization**
```python
# Tối ưu quảng cáo bằng AI
POST /api/v1/ai/optimize-ad
```
- AI-generated slogans
- Target audience analysis
- Performance predictions

---

## 🐛 **Troubleshooting**

### **Lỗi: API Key không hợp lệ**
```
Error: Invalid API key
```
**Giải pháp:**
1. Kiểm tra API Key đã copy đúng chưa
2. Không có khoảng trắng thừa
3. API Key còn active

### **Lỗi: Quota exceeded**
```
Error: Quota exceeded for quota metric
```
**Giải pháp:**
- Đợi 1 phút (rate limit)
- Hoặc đợi đến ngày mai (daily limit)
- Upgrade lên paid tier

### **Lỗi: Module not found**
```
FutureWarning: google.generativeai package
```
**Giải pháp:**
- Warning này có thể ignore
- Hoặc upgrade: `pip install google-generativeai --upgrade`

---

## 📝 **Environment Variables Summary**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_AI_API_KEY` | ✅ Yes | None | Google Gemini API Key |
| `AI_PROVIDER` | No | `google_ai` | AI provider to use |
| `OPENAI_API_KEY` | No | None | OpenAI API Key (optional) |

---

## 🎓 **Resources**

- **Google AI Studio**: https://makersuite.google.com/
- **API Documentation**: https://ai.google.dev/docs
- **Gemini Models**: https://ai.google.dev/models/gemini
- **Pricing**: https://ai.google.dev/pricing

---

## 🚀 **Quick Start Command**

```bash
# 1. Get API Key
open https://makersuite.google.com/app/apikey

# 2. Add to .env
echo "GOOGLE_AI_API_KEY=YOUR_API_KEY" >> backend_api/.env
echo "AI_PROVIDER=google_ai" >> backend_api/.env

# 3. Start Backend
START.bat → [3] Run Backend

# 4. Test
curl http://localhost:8000/api/v1/ai/status
```

---

**Done! Your AI Agent is ready to use with Google Gemini!** 🎉
