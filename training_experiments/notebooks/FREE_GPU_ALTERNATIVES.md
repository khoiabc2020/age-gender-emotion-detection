# 🆓 Free GPU Alternatives for Training

## Khi hết GPU miễn phí trên Colab, bạn có các lựa chọn sau:

---

## 🥇 1. KAGGLE NOTEBOOKS (Khuyến nghị nhất)

### ✅ Ưu điểm:
- **30 giờ GPU/tuần** (nhiều hơn Colab 2.5 lần)
- **GPU: T4 hoặc P100** (P100 mạnh hơn T4)
- **Ổn định hơn** - không bị disconnect thường xuyên
- **Datasets có sẵn** - FER2013, UTKFace đã có trên Kaggle
- **TPU miễn phí** (nếu cần)
- **100% miễn phí**, không cần thẻ tín dụng

### 📋 Cách dùng:
1. Tạo tài khoản: https://www.kaggle.com/
2. Tạo notebook mới: https://www.kaggle.com/code
3. Settings → Accelerator → **GPU T4 x2** hoặc **GPU P100**
4. Settings → Internet → **ON**
5. Add Input → Add datasets (FER2013, UTKFace)
6. Copy code từ file `KAGGLE_TRAINING_GUIDE.md`

### ⏱️ GPU Quota:
- **30 hours/week** (reset mỗi tuần)
- Monitor tại: https://www.kaggle.com/settings

### 📖 Hướng dẫn chi tiết:
→ Xem file: `KAGGLE_TRAINING_GUIDE.md`

---

## 🥈 2. LIGHTNING.AI (Khá tốt)

### ✅ Ưu điểm:
- **22 giờ GPU/tháng** miễn phí
- **GPU: T4**
- **Persistent storage** (lưu dữ liệu giữa các sessions)
- **VSCode interface** (quen thuộc)
- **SSH access**

### ❌ Nhược điểm:
- Quota theo tháng (không phải tuần)
- Cần verify email

### 📋 Cách dùng:
1. Tạo tài khoản: https://lightning.ai/
2. Tạo Studio mới
3. Chọn **GPU (Free)**
4. Clone repo:
   ```bash
   git clone https://github.com/khoiabc2020/age-gender-emotion-detection.git
   cd age-gender-emotion-detection/training_experiments
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements_production.txt
   ```
6. Train:
   ```bash
   python train_production.py --epochs 100 --batch_size 64
   ```

### 🔗 Link:
https://lightning.ai/

---

## 🥉 3. PAPERSPACE GRADIENT (Giới hạn nhưng OK)

### ✅ Ưu điểm:
- **Free tier với GPU** (giới hạn)
- **6 giờ runtime**
- **Jupyter Notebook interface**
- **Gradient Notebooks** - tương tự Colab

### ❌ Nhược điểm:
- Chỉ 6 giờ/session
- Cần verify thẻ tín dụng (không charge)
- Queue time có thể lâu

### 📋 Cách dùng:
1. Tạo tài khoản: https://console.paperspace.com/signup
2. Verify email + card (không charge)
3. Tạo notebook: Gradient → Notebooks → Create
4. Chọn **Free-GPU** runtime
5. Clone và train tương tự Colab

### 🔗 Link:
https://gradient.run/notebooks

---

## 4. GOOGLE COLAB PRO (Paid nhưng rẻ)

### 💰 Giá:
- **Colab Pro**: $9.99/tháng
- **Colab Pro+**: $49.99/tháng

### ✅ Ưu điểm (Pro):
- **100 compute units/tháng** (~40-50 giờ GPU)
- **GPU: T4, V100** (có thể chọn)
- **Background execution** (không bị disconnect)
- **32GB RAM** (vs 12GB free)

### 📋 Nếu budget cho phép:
→ https://colab.research.google.com/signup

---

## 5. VAST.AI (Rẻ nhất nếu cần thuê)

### 💰 Giá:
- **$0.10 - $0.30/giờ** (RTX 3090, A5000)
- **~$2-3 cho 10 giờ** training

### ✅ Ưu điểm:
- Rất rẻ so với cloud khác
- Nhiều loại GPU (RTX 3090, A6000, etc.)
- Pay-as-you-go

### ❌ Nhược điểm:
- Phải trả tiền (nhưng rất ít)
- Setup phức tạp hơn

### 📋 Cách dùng:
1. Tạo tài khoản: https://vast.ai/
2. Nạp $5-10
3. Tìm instance với GPU tốt, giá rẻ
4. SSH vào và train

### 🔗 Link:
https://vast.ai/

---

## 6. SAGEMAKER STUDIO LAB (AWS Free)

### ✅ Ưu điểm:
- **15 giờ GPU/session** miễn phí
- **GPU: T4**
- **Persistent storage**
- Không cần thẻ tín dụng

### ❌ Nhược điểm:
- Cần request access (chờ vài ngày)
- Interface AWS (hơi khó)

### 📋 Cách dùng:
1. Request access: https://studiolab.sagemaker.aws/
2. Đợi email approve (2-5 ngày)
3. Login và tạo notebook
4. Chọn GPU runtime
5. Clone repo và train

### 🔗 Link:
https://studiolab.sagemaker.aws/

---

## 7. TRAIN TRÊN LOCAL (CPU) - Chậm nhưng FREE

### ✅ Ưu điểm:
- **Hoàn toàn miễn phí**
- **Không giới hạn thời gian**
- **Full control**

### ❌ Nhược điểm:
- **RẤT CHẬM** (10-20x chậm hơn GPU)
- Training có thể mất **100-200 giờ** (4-8 ngày)

### 📋 Cách dùng:
```bash
cd training_experiments

# Giảm epochs và batch size cho CPU
python train_production.py \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.0001 \
    --device cpu
```

### 💡 Tips để train trên CPU nhanh hơn:
1. Giảm batch size: `--batch_size 8`
2. Giảm epochs: `--epochs 20`
3. Dùng model nhỏ hơn: MobileNet thay vì EfficientNet
4. Train overnight
5. Dùng PyTorch với MKL optimization

---

## 8. RUNPOD.IO (Pay-as-you-go, rẻ)

### 💰 Giá:
- **$0.20 - $0.40/giờ** (RTX 3090, A5000)
- **~$3-5 cho 10 giờ**

### ✅ Ưu điểm:
- Rẻ, stable
- Nhiều GPU options
- Jupyter interface
- Quick setup

### 🔗 Link:
https://www.runpod.io/

---

## 📊 So sánh tổng quan:

| Platform | GPU Time | GPU Type | Cost | Stability | Difficulty |
|----------|----------|----------|------|-----------|------------|
| **Kaggle** ⭐ | 30h/week | T4, P100 | FREE | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Lightning.ai** | 22h/month | T4 | FREE | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Paperspace | 6h/session | Free GPU | FREE | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Colab Free | 12h/day | T4 | FREE | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Sagemaker Lab | 15h/session | T4 | FREE | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Vast.ai | Unlimited | Various | $0.1-0.3/h | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Colab Pro | 100 CU/month | T4, V100 | $9.99/mo | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Local CPU | Unlimited | CPU | FREE | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 Khuyến nghị theo tình huống:

### 🆓 **Nếu cần hoàn toàn miễn phí:**
1. **Kaggle Notebooks** (30h/week) - Tốt nhất
2. **Lightning.ai** (22h/month) - Backup option
3. **Paperspace Gradient** (6h/session) - Nếu train ngắn

### 💰 **Nếu có budget nhỏ ($5-10):**
1. **Vast.ai** - Rẻ nhất, flexible
2. **Runpod.io** - Stable, easy setup
3. **Colab Pro** - Nếu dùng thường xuyên

### ⏰ **Nếu không vội:**
1. **Train trên local CPU** - Chậm nhưng free
2. Chờ Colab quota reset
3. Request Sagemaker Studio Lab

---

## 🚀 Lộ trình thực tế:

### **Tuần 1:**
- Train trên **Kaggle** (30h GPU)
- Nếu chưa xong, tiếp tục tuần sau

### **Tuần 2:**
- Tiếp tục **Kaggle** (30h nữa)
- Hoặc dùng **Lightning.ai** (22h)

### **Nếu vẫn chưa đủ:**
- Thuê **Vast.ai** ~$3 cho 10 giờ để hoàn thành
- Hoặc train trên local CPU overnight

---

## 📝 Tóm tắt:

**Giải pháp TỐT NHẤT cho bạn:**

1. **Ngay bây giờ**: Dùng **Kaggle Notebooks** (30h/week, miễn phí)
2. **Backup**: **Lightning.ai** (22h/month, miễn phí)
3. **Nếu gấp**: Thuê **Vast.ai** ($2-3 cho 10h)

**→ Tôi khuyến nghị dùng Kaggle trước, xem hướng dẫn trong `KAGGLE_TRAINING_GUIDE.md`**

---

**Chúc bạn training thành công! 🎉**
