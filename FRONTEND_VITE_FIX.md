# Frontend Vite Installation Issue - Workaround

## 🔴 **VẤN ĐỀ:**

Vite không được cài vào `node_modules` mặc dù có trong `package.json`.

## ✅ **GIẢI PHÁP TẠM THỜI:**

### **Option 1: Cài Vite Global (Recommended)**

```bash
npm install -g vite@5.4.21
```

Sau đó chạy:
```bash
cd dashboard
vite
```

### **Option 2: Dùng npx với cache**

```bash
cd dashboard
npx --yes vite@5.4.21
```

### **Option 3: Manual Install**

```bash
cd dashboard
mkdir -p node_modules/vite
cd node_modules/vite
npm install vite@5.4.21 --save
cd ../../..
npm run dev
```

---

## 🔧 **FIX PERMANENT:**

Có thể do npm version hoặc cache issue. Thử:

```bash
# 1. Xóa hoàn toàn
cd dashboard
Remove-Item -Recurse -Force node_modules,package-lock.json,.vite-temp

# 2. Clear cache
npm cache clean --force

# 3. Cài lại
npm install --legacy-peer-deps --verbose

# 4. Verify
npm list vite
```

---

## 📝 **TẠM THỜI:**

Frontend có thể chạy được với `npx vite` nếu vite được cài global hoặc có trong PATH.

Backend và Edge App đã chạy OK!
