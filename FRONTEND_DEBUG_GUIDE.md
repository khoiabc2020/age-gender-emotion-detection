# 🔍 DEBUG FRONTEND - HƯỚNG DẪN KIỂM TRA

## ✅ **ĐÃ FIX:**

### **1. Error Boundary**
- ✅ Thêm Error Boundary để catch lỗi React
- ✅ Hiển thị lỗi rõ ràng thay vì trang trắng
- ✅ Có nút "Tải lại trang"

### **2. Error Handling**
- ✅ Safe localStorage access trong authSlice
- ✅ Try-catch trong main.jsx
- ✅ Console logging để debug

### **3. Frontend Restart**
- ✅ Khởi động lại trong window mới
- ✅ Port 3000 đã được clear

---

## 🔍 **CÁCH KIỂM TRA:**

### **Bước 1: Kiểm tra Frontend có chạy không**
```bash
# Mở browser console (F12)
# Xem có lỗi gì không
```

### **Bước 2: Kiểm tra Network**
```bash
# F12 → Network tab
# Xem có request nào fail không
# Kiểm tra main.jsx có load không
```

### **Bước 3: Kiểm tra Console**
```bash
# F12 → Console tab
# Tìm message: "✅ React app rendered successfully!"
# Nếu có lỗi, sẽ hiển thị chi tiết
```

### **Bước 4: Kiểm tra Redux Store**
```bash
# F12 → Console
# Gõ: window.__REDUX_DEVTOOLS_EXTENSION__ 
# Hoặc cài Redux DevTools extension
```

---

## 🚀 **CÁCH CHẠY LẠI:**

### **Option 1: Dùng START.bat**
```bash
START.bat → [4] Run Frontend
```

### **Option 2: Manual**
```bash
cd dashboard
npm run dev
```

### **Option 3: Clear cache và chạy lại**
```bash
cd dashboard
rm -rf node_modules/.vite
npm run dev
```

---

## 📋 **CHECKLIST:**

- [ ] Frontend đang chạy trên port 3000
- [ ] Browser console không có lỗi
- [ ] Network tab thấy main.jsx load thành công
- [ ] Có message "✅ React app rendered successfully!" trong console
- [ ] Trang hiển thị Login page hoặc Dashboard (không phải trắng)

---

## 🐛 **NẾU VẪN TRẮNG:**

### **1. Kiểm tra Browser Console**
- Mở F12 → Console
- Copy toàn bộ lỗi và gửi cho tôi

### **2. Kiểm tra Network**
- F12 → Network
- Xem file nào fail (màu đỏ)
- Copy URL và status code

### **3. Clear Browser Cache**
- Ctrl + Shift + Delete
- Clear cache và cookies
- Refresh (Ctrl + F5)

### **4. Test với Test Page**
```bash
# Tạm thời đổi trong index.html:
# <script type="module" src="/src/test.jsx"></script>
# Nếu test.jsx hiển thị → Vấn đề ở App.jsx
# Nếu test.jsx cũng trắng → Vấn đề ở Vite/React setup
```

---

## ✅ **KẾT QUẢ MONG ĐỢI:**

Khi truy cập http://localhost:3000:
- **Nếu chưa login**: Hiển thị Login page
- **Nếu đã login**: Hiển thị Dashboard
- **Nếu có lỗi**: Hiển thị Error Boundary với thông báo lỗi

**KHÔNG BAO GIỜ** nên thấy trang trắng hoàn toàn!

---

## 📞 **NẾU VẪN LỖI:**

Gửi cho tôi:
1. Screenshot browser console (F12 → Console)
2. Screenshot Network tab (F12 → Network)
3. Toàn bộ output từ terminal khi chạy `npm run dev`
