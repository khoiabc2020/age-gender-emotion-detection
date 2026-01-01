# ⚠️ Python Version Compatibility Issue

## 🔴 **LỖI: ONNX Runtime không hỗ trợ Python 3.13+**

---

## 📋 **VẤN ĐỀ**

```
ERROR: Could not find a version that satisfies the requirement onnxruntime>=1.16.0
ERROR: No matching distribution found for onnxruntime>=1.16.0
```

### **Nguyên nhân:**
- 🔴 **Python 3.13/3.14** quá mới
- 🔴 **ONNX Runtime** chưa release wheel cho Python 3.13+
- 🔴 **Edge AI App** cần ONNX Runtime để chạy model

---

## ✅ **GIẢI PHÁP**

### **Option 1: Dùng Python 3.12 (RECOMMENDED)** ⭐⭐⭐

#### **Bước 1: Tải Python 3.12**
- 📥 Download: https://www.python.org/downloads/release/python-3120/
- Chọn: **Windows installer (64-bit)**
- Cài đặt: ✅ Add to PATH

#### **Bước 2: Cài Dependencies**
```bash
# Mở terminal mới (để load Python 3.12)
python --version
# Should show: Python 3.12.x

# Chạy installer
INSTALL_EDGE_AI.bat
```

#### **Bước 3: Chạy App**
```bash
run_app\run_edge.bat
```

---

### **Option 2: Dùng Virtual Environment với Python 3.12**

```bash
# Tạo venv với Python 3.12
py -3.12 -m venv venv_edge

# Activate
venv_edge\Scripts\activate

# Cài dependencies
cd ai_edge_app
pip install -r requirements.txt

# Chạy app
python main.py
```

---

### **Option 3: Docker (EASIEST)** 🐳

```bash
# Không cần quan tâm Python version
# Docker tự động dùng Python 3.11

# Chạy tất cả
docker-compose up -d

# Chỉ Edge AI
docker-compose up ai-edge-app
```

---

### **Option 4: Build ONNX Runtime từ source (ADVANCED)** 🔧

```bash
# Chỉ dành cho advanced users
# Follow: https://onnxruntime.ai/docs/build/

git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime
# ... build instructions
```

---

## 🔍 **KIỂM TRA PYTHON VERSION**

```bash
# Kiểm tra version hiện tại
python --version

# Kiểm tra tất cả Python versions
py --list

# Dùng Python 3.12 cụ thể
py -3.12 --version
```

---

## 📊 **COMPATIBILITY MATRIX**

| Python Version | ONNX Runtime | Status |
|----------------|--------------|--------|
| 3.8 | ✅ 1.15.0+ | Hoàn toàn OK |
| 3.9 | ✅ 1.15.0+ | Hoàn toàn OK |
| 3.10 | ✅ 1.15.0+ | Hoàn toàn OK |
| 3.11 | ✅ 1.15.0+ | Hoàn toàn OK |
| 3.12 | ✅ 1.16.0+ | Hoàn toàn OK |
| 3.13 | ⚠️ Limited | Experimental |
| 3.14 | ❌ None | Không hỗ trợ |

---

## 🎯 **KHUYẾN NGHỊ**

### **Cho Production:**
- ✅ **Python 3.11** hoặc **3.12**
- ✅ Docker (tự động dùng Python 3.11)

### **Cho Development:**
- ✅ Python 3.12 (latest stable với ONNX support)
- ✅ Virtual environment riêng cho project

### **Không nên dùng:**
- ❌ Python 3.13+ (too new)
- ❌ Python 3.7 trở xuống (deprecated)

---

## 🔧 **QUICK FIX**

### **Nếu bạn đang dùng Python 3.13+:**

```bash
# 1. Tải Python 3.12
https://www.python.org/downloads/release/python-3120/

# 2. Cài đặt (tick "Add to PATH")

# 3. Mở terminal MỚI

# 4. Kiểm tra
python --version
# Nên thấy: Python 3.12.x

# 5. Cài dependencies
INSTALL_EDGE_AI.bat

# 6. Chạy
run_app\run_edge.bat
```

---

## ❓ **FAQ**

**Q: Tôi có thể dùng nhiều Python versions cùng lúc không?**  
A: Được! Dùng `py -3.12` để chỉ định version cụ thể.

**Q: Có cần gỡ Python cũ không?**  
A: Không cần, có thể giữ cả 2 versions.

**Q: Docker có dễ hơn không?**  
A: Có! Docker tự động lo hết vấn đề về dependencies.

**Q: Tôi chỉ chạy Backend + Frontend thôi, có cần fix không?**  
A: Không! Lỗi này chỉ ảnh hưởng Edge AI App.

---

## 📝 **FILES LIÊN QUAN**

- `ai_edge_app/requirements.txt` - Dependencies list
- `INSTALL_EDGE_AI.bat` - Edge AI installer (với version check)
- `INSTALL_DEPENDENCIES.bat` - Full installer (tất cả components)
- `docker-compose.yml` - Docker setup (Python 3.11)

---

## 🚀 **RECOMMENDED WORKFLOW**

```bash
# 1. Clone repo
git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git
cd age-gender-emotion-detection

# 2. Kiểm tra Python
python --version
# Nếu không phải 3.11 hoặc 3.12 → Tải Python 3.12

# 3. Cài dependencies
INSTALL_DEPENDENCIES.bat

# 4. Chạy app
QUICK_START.bat
```

---

**TÓM LẠI: Dùng Python 3.12 hoặc Docker là đơn giản nhất!** 🎯
