# **📘 ĐỒ ÁN TỐT NGHIỆP: HỆ THỐNG SMART RETAIL ANALYTICS & TARGETED ADS**

**Tên đề tài:** Nghiên cứu và Xây dựng Hệ thống Nhận diện Khách hàng & Đề xuất Quảng cáo Cá nhân hóa sử dụng Deep Learning và Edge Computing.

**Phiên bản:** 3.1.0 (Final Architecture - Full Stack)

**Sinh viên thực hiện:** [Tên Của Bạn]

**GV Hướng dẫn:** [Tên GV]

## **📑 MỤC LỤC**

1. [Giới Thiệu & Đóng Góp Của Đề Tài](#1-giới-thiệu--đóng-góp-của-đề-tài)
2. [Cơ Sở Lý Thuyết & Đạo Đức AI](#2-cơ-sở-lý-thuyết--đạo-đức-ai)
3. [Kiến Trúc Hệ Thống (Microservices & Async)](#3-kiến-trúc-hệ-thống-microservices--async)
4. [Chi Tiết Kỹ Thuật & Tối Ưu Hóa Model](#4-chi-tiết-kỹ-thuật--tối-ưu-hóa-model)
5. [Backend, Database & Security](#5-backend-database--security)
6. [Frontend Dashboard & Trải Nghiệm Người Dùng](#6-frontend-dashboard--trải-nghiệm-người-dùng)
7. [Logic Đề Xuất Quảng Cáo (Hybrid)](#7-logic-đề-xuất-quảng-cáo-hybrid)
8. [Đánh Giá & Kết Quả Thử Nghiệm](#8-đánh-giá--kết-quả-thử-nghiệm)
9. [Cài Đặt & Triển Khai (DevOps)](#9-cài-đặt--triển-khai-devops)
10. [Quản Lý Dự Án & Rủi Ro](#10-quản-lý-dự-án--rủi-ro)

## **1. GIỚI THIỆU & ĐÓNG GÓP CỦA ĐỀ TÀI**

### **1.1. Đặt vấn đề**

Trong kỷ nguyên bán lẻ 4.0, các biển quảng cáo truyền thống (Static Signage) đang lãng phí ngân sách vì không nhắm đúng đối tượng. Doanh nghiệp cần chuyển đổi sang **Digital Signage** thông minh, có khả năng "nhìn" và "hiểu" khách hàng để tối ưu hóa trải nghiệm mua sắm (Customer Experience).

### **1.2. Mục tiêu đồ án**

Xây dựng hệ thống trọn vẹn từ Edge (Biên) đến Cloud (Đám mây):

1. **Thu thập & Phân tích:** Nhận diện thuộc tính nhân khẩu học (Tuổi, Giới tính) và tâm lý (Cảm xúc) realtime.  
2. **Ra quyết định:** Gợi ý nội dung quảng cáo động (Dynamic Content) dưới 200ms.  
3. **Báo cáo:** Cung cấp Dashboard phân tích hành vi người tiêu dùng (Consumer Behavior).

### **1.3. Tính mới & Đóng góp của đề tài (Điểm nhấn)**

* **Privacy-First Design:** Hệ thống chỉ trích xuất đặc trưng số (vector/metadata), **không lưu trữ hình ảnh khuôn mặt gốc**, đảm bảo quyền riêng tư.  
* **Edge Optimization:** Tối ưu hóa mô hình Deep Learning để chạy mượt mà trên thiết bị cấu hình thấp (Laptop không GPU hoặc Jetson Nano) bằng TensorRT/ONNX.  
* **Scoring System:** Thuật toán xếp hạng quảng cáo đa tiêu chí (Tuổi + Giới tính + Cảm xúc + Lịch sử hiển thị).

## **2. CƠ SỞ LÝ THUYẾT & ĐẠO ĐỨC AI**

### **2.1. Dataset & Tiền xử lý nâng cao**

* **UTKFace & FairFace:** Sử dụng kết hợp để cân bằng dữ liệu chủng tộc, tránh bias (thiên kiến) mô hình.  
* **Data Augmentation:** Sử dụng thư viện Albumentations để tạo các biến thể khó: Motion Blur (mờ do chuyển động), Low Light (ánh sáng yếu), Rain/Fog (giả lập môi trường).

### **2.2. Vấn đề Đạo đức & Quyền riêng tư (Ethical AI)**

* **Cơ chế:** Ngay sau khi Model dự đoán xong Tuổi/Giới tính, hình ảnh khuôn mặt trong RAM sẽ bị hủy (Discard).  
* **Lưu trữ:** Chỉ lưu trữ log dạng văn bản: Time: 10:00, Age: 25, Gender: Male, Emotion: Happy.  
* **Tuân thủ:** Thiết kế hướng tới việc tuân thủ cơ bản GDPR (nếu triển khai thực tế).

## **3. KIẾN TRÚC HỆ THỐNG (MICROSERVICES & ASYNC)**

Nâng cấp kiến trúc sử dụng **Message Queue** để hệ thống không bị "treo" khi lượng khách quá đông.

```
graph TD
    subgraph "Edge Layer (Client Store)"
        CAM[Camera Input] --> DET[Face Detector (RetinaFace)]
        DET --> TRACK[Tracker (DeepSORT)]
        TRACK --> AI[Attribute Model (EfficientNet)]
        AI --> UI[Ads Display Screen]
        AI -.->|Async Data Push| MQTT[MQTT Broker / Redis]
    end

    subgraph "Cloud/Server Layer"
        MQTT --> WORKER[Background Worker]
        WORKER --> DB[(PostgreSQL TimescaleDB)]
        DB --> API[FastAPI Server]
        API --> DASH[Analytics Dashboard]
    end
```

## **4. CHI TIẾT KỸ THUẬT & TỐI ƯU HÓA MODEL**

### **4.1. Kiến trúc mạng đa nhiệm (Multi-task Learning)**

Thay vì chạy 3 model riêng lẻ (nặng máy), xây dựng 1 Backbone chia sẻ chung:

* **Input:** 224x224x3 Image.  
* **Shared Backbone:** MobileNetV3-Large hoặc EfficientNet-B0 (nhẹ và nhanh).  
* **Heads:**  
  * Head 1 (Gender): Classification (Binary).  
  * Head 2 (Age): Regression (L1 Loss).  
  * Head 3 (Emotion): Classification (CrossEntropy).

### **4.2. Tối ưu hóa Edge Computing (Kỹ thuật cao)**

Đây là phần giúp đồ án đạt điểm tối đa về mặt kỹ thuật:

* **Model Pruning:** Cắt tỉa các nơ-ron ít quan trọng để giảm kích thước model.  
* **Quantization (Lượng tử hóa):** Chuyển đổi trọng số model từ Float32 sang Int8 hoặc Float16.  
  * *Kết quả:* Giảm kích thước model 4 lần, tăng tốc độ inference 2-3 lần mà độ chính xác giảm không đáng kể (<1%).  
* **Runtime:** Sử dụng **ONNX Runtime** hoặc **OpenVINO** để chạy model thay vì PyTorch thuần.

## **5. BACKEND, DATABASE & SECURITY**

### **5.1. Database tối ưu cho Time-series**

Sử dụng **PostgreSQL** kết hợp logic Time-series để truy vấn nhanh các câu hỏi như: *"Vào lúc 9-10h sáng, độ tuổi trung bình khách hàng là bao nhiêu?"*.

**Schema mở rộng:**

* sessions: Lưu phiên khách hàng (ID, StartTime, EndTime).  
* interactions: Lưu từng khoảnh khắc cảm xúc thay đổi trong phiên.

### **5.2. Bảo mật hệ thống (Security)**

* **API Authentication:** Sử dụng JWT (JSON Web Token) để bảo vệ API Dashboard.  
* **Edge Authentication:** Mỗi thiết bị Edge (Camera) có một Device_Key riêng để gửi dữ liệu về Server, tránh giả mạo dữ liệu.

## **6. FRONTEND DASHBOARD & TRẢI NGHIỆM NGƯỜI DÙNG**

Phần này mô tả giao diện quản trị (Dashboard) dành cho người quản lý hệ thống.

### **6.1. Công nghệ Frontend (Tech Stack)**

* **Framework:** ReactJS (hoặc Next.js) - Lựa chọn chuẩn công nghiệp để đảm bảo hiệu năng cao, khả năng tương tác mượt mà và dễ dàng mở rộng (Scalability).  
* **State Management:** Redux Toolkit hoặc React Query để quản lý trạng thái ứng dụng và đồng bộ dữ liệu realtime từ API.  
* **Styling:** Tailwind CSS kết hợp với Ant Design hoặc Material UI để xây dựng giao diện hiện đại, chuẩn Responsive (tương thích mobile/tablet).  
* **Data Visualization:** Recharts hoặc Chart.js để vẽ các biểu đồ phân tích dữ liệu trực quan.

### **6.2. Các chức năng chính của Dashboard**

#### **Real-time Monitor (Giám sát thời gian thực):**

* Hiển thị trạng thái hoạt động của các Camera (Online/Offline).  
* Xem luồng dữ liệu log (Stream logs) đang đổ về từ các Edge device thông qua WebSocket.

#### **Analytics Reports (Báo cáo phân tích):**

* **Biểu đồ đường:** Lưu lượng khách hàng theo khung giờ trong ngày.  
* **Biểu đồ tròn:** Tỷ lệ Nam/Nữ, Phân bố nhóm tuổi.  
* **Biểu đồ nhiệt (Heatmap):** Cảm xúc trung bình của khách hàng theo các ngày trong tuần.

#### **Ads Management (Quản lý quảng cáo):**

* Giao diện Upload video/hình ảnh quảng cáo.  
* Cấu hình luật hiển thị (Targeting Rules): Ví dụ "Kéo thả video A vào nhóm khách hàng Nam - 18-25 tuổi".

### **6.3. Trải nghiệm người dùng (UX Flow)**

* **Admin User:** Đăng nhập an toàn -> Xem Tổng quan hệ thống (Dashboard) để nắm tình hình -> Vào Cấu hình Quảng cáo để điều chỉnh chiến dịch -> Xuất báo cáo (Export Excel/PDF) để báo cáo cấp trên.  
* **End-User (Khách hàng tại cửa hàng):** Bước vào vùng nhận diện -> Màn hình quảng cáo thay đổi nội dung tức thì (Latency < 200ms) -> Nội dung phù hợp gây chú ý -> Tăng trải nghiệm mua sắm cá nhân hóa.

## **7. LOGIC ĐỀ XUẤT QUẢNG CÁO (HYBRID)**

Nâng cấp thuật toán đề xuất để thông minh hơn.

**Quy trình 3 bước:**

1. **Filtering (Lọc):** Loại bỏ quảng cáo không phù hợp độ tuổi/giới tính.  
2. **Scoring (Chấm điểm):**  
   * *Context Score:* Điểm theo giờ (VD: Sáng ưu tiên Cafe, Trưa ưu tiên Cơm).  
   * *Emotion Score:* Khách vui -> Gợi ý sản phẩm cao cấp; Khách buồn -> Gợi ý dịch vụ giải trí.  
3. **Exploration (Khám phá):** Đôi khi hiển thị ngẫu nhiên 1 quảng cáo mới (10% tỉ lệ) để đo lường phản ứng (A/B Testing).

## **8. ĐÁNH GIÁ & KẾT QUẢ THỬ NGHIỆM**

### **8.1. Định lượng (Quantitative)**

* **Độ chính xác (Accuracy):**  
  * Giới tính: > 92%.  
  * Tuổi (MAE): < 4.5 năm.  
  * Cảm xúc: > 75% (trên tập test FER2013).  
* **Hiệu năng (Performance):**  
  * Latency (Độ trễ): < 150ms trên Laptop Core i5 (CPU).  
  * FPS: Duy trì ổn định 20-25 FPS.

### **8.2. Định tính (Qualitative)**

* Khả năng hoạt động ổn định khi có 3-4 người cùng lúc.  
* Khả năng phục hồi (Recovery) khi khách quay mặt đi rồi quay lại (nhờ DeepSORT).

## **9. CÀI ĐẶT & TRIỂN KHAI (DEVOPS)**

Sử dụng Docker để chuẩn hóa môi trường, giúp hội đồng dễ dàng chấm điểm demo.

### **9.1. Cấu trúc Project Chi Tiết (Full Directory Tree)**

Đây là cấu trúc thư mục chuẩn Enterprise, phân tách rõ ràng giữa Edge (AI), Backend (API), Frontend (Dashboard) và Data.

```
Smart-Retail-Ads/
│
├── 📂 ai_edge_app/             # (CLIENT) Ứng dụng chạy trên Camera/Laptop
│   ├── 📂 configs/             # File cấu hình (JSON/YAML)
│   │   ├── camera_config.json
│   │   └── ads_rules.json      # Luật quảng cáo (offline fallback)
│   ├── 📂 models/              # Chứa weights đã tối ưu
│   │   ├── retinaface_mnet.onnx
│   │   └── multitask_efficientnet_int8.onnx
│   ├── 📂 src/
│   │   ├── 📂 detectors/       # Module RetinaFace
│   │   ├── 📂 trackers/        # Module DeepSORT
│   │   ├── 📂 classifiers/     # Module Age/Gender/Emotion
│   │   ├── 📂 ads_engine/      # Logic tính điểm quảng cáo
│   │   └── 📂 utils/           # MQTT Client, Logger
│   ├── main.py                 # Entry point chạy Camera
│   ├── Dockerfile              # Container hóa Edge App
│   └── requirements.txt
│
├── 📂 backend_api/             # (SERVER) Xử lý logic nghiệp vụ & DB
│   ├── 📂 app/
│   │   ├── 📂 api/             # Endpoints (GET, POST)
│   │   ├── 📂 core/            # Config, Security (JWT)
│   │   ├── 📂 db/              # SQLAlchemy models & CRUD
│   │   ├── 📂 schemas/         # Pydantic models (Data validation)
│   │   └── main.py             # FastAPI App
│   ├── Dockerfile
│   └── requirements.txt
│
├── 📂 dashboard/               # (FRONTEND) Giao diện quản trị
│   ├── 📂 src/
│   │   ├── 📂 components/      # UI Components (Chart, Table, Card)
│   │   ├── 📂 pages/           # Dashboard, Login, Settings
│   │   ├── 📂 services/        # API integration (Axios)
│   │   ├── 📂 store/           # Redux slices
│   │   └── App.js
│   ├── public/
│   ├── Dockerfile
│   └── package.json            # React dependencies
│
├── 📂 database/                # (STORAGE)
│   └── init.sql                 # Script tạo bảng ban đầu
│
├── 📂 training_experiments/    # (RESEARCH) Nơi huấn luyện model
│   ├── 📂 data/                # Dataset (UTKFace, FER2013)
│   ├── 📂 notebooks/           # Jupyter Notebooks (EDA, Training)
│   └── 📂 scripts/             # Script convert model sang ONNX
│
├── docker-compose.yml          # File khởi chạy toàn bộ hệ thống
└── README.md                   # Tài liệu hướng dẫn
```

### **9.2. Continuous Integration (CI) - Optional**

Thiết lập GitHub Actions đơn giản để kiểm tra lỗi code (Linting) mỗi khi push code lên repo.

## **10. QUẢN LÝ DỰ ÁN & RỦI RO**

Phần này thể hiện tư duy của một kỹ sư trưởng dự án.

### **10.1. Quản lý rủi ro (Risk Management)**

| Rủi ro | Mức độ | Giải pháp |
| :---- | :---- | :---- |
| Ánh sáng yếu, ngược sáng | Cao | Dùng thuật toán cân bằng sáng (CLAHE) trước khi đưa vào model. |
| Khách đeo khẩu trang | Trung bình | Training bổ sung dữ liệu đeo khẩu trang; Tập trung đặc trưng vùng mắt/trán. |
| Độ trễ mạng cao | Thấp | Xử lý logic quảng cáo Offline tại Edge (Local), chỉ gửi log về Server sau (Async). |

### **10.2. Kế hoạch phát triển (Future Work)**

* Tích hợp nhận diện khách hàng VIP (Face Recognition) khi được sự đồng ý.  
* Phân tích hành vi nhìn (Gaze estimation): Biết khách đang nhìn vào góc nào của màn hình.

**Kết luận:** Đồ án này không chỉ dừng lại ở việc áp dụng Deep Learning mà còn giải quyết bài toán hệ thống tổng thể (System Design), tối ưu hóa hiệu năng thực tế (Deployment) và quan tâm đến khía cạnh đạo đức (Ethics). Đây là tiêu chuẩn của một đồ án kỹ sư chất lượng cao.
