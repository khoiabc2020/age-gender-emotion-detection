# Hướng dẫn Setup và Chạy Dự án

## Bước 1: Cài đặt Dependencies

### Option 1: Sử dụng Docker (Khuyến nghị)

```bash
# Đảm bảo đã cài Docker và Docker Compose
docker --version
docker-compose --version

# Khởi động toàn bộ hệ thống
docker-compose up -d

# Kiểm tra các service đang chạy
docker-compose ps

# Xem logs
docker-compose logs -f backend
docker-compose logs -f dashboard
```

### Option 2: Cài đặt Local

#### 1. Cài đặt PostgreSQL

```bash
# Windows (sử dụng Chocolatey hoặc download từ postgresql.org)
# macOS
brew install postgresql
brew services start postgresql

# Linux
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

Tạo database:
```sql
CREATE DATABASE retail_analytics;
```

#### 2. Cài đặt Python Dependencies

```bash
# Tạo virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Cài đặt dependencies cho từng module
cd backend_api
pip install -r requirements.txt

cd ../dashboard
pip install -r requirements.txt

cd ../ai_edge_app
pip install -r requirements.txt
```

#### 3. Cài đặt MQTT Broker (Mosquitto)

```bash
# Windows: Download từ https://mosquitto.org/download/
# macOS
brew install mosquitto
brew services start mosquitto

# Linux
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

## Bước 2: Cấu hình

### Backend API

Tạo file `backend_api/.env`:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/retail_analytics
SECRET_KEY=your-secret-key-here-change-in-production
MQTT_BROKER=localhost
MQTT_PORT=1883
DEBUG=true
```

### Edge App

Chỉnh sửa `ai_edge_app/configs/camera_config.json` nếu cần:
- Thay đổi camera source (0 cho webcam mặc định)
- Cấu hình MQTT broker nếu không dùng localhost

## Bước 3: Khởi tạo Database

```bash
# Nếu dùng Docker, database sẽ tự động khởi tạo
# Nếu chạy local, chạy script init:
psql -U postgres -d retail_analytics -f database/init.sql
```

## Bước 4: Thêm Model Files

Để chạy Edge App, cần có các model files:

1. **RetinaFace model**: `ai_edge_app/models/retinaface_mnet.onnx`
2. **Multi-task classifier**: `ai_edge_app/models/multitask_efficientnet_int8.onnx`

Các model này cần được train và convert từ PyTorch sang ONNX format.

## Bước 5: Chạy Ứng dụng

### Với Docker:

```bash
# Tất cả services đã chạy với docker-compose up
# Truy cập:
# - Backend API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

### Chạy Local:

#### Terminal 1: Backend API
```bash
cd backend_api
uvicorn app.main:app --reload
```

#### Terminal 2: MQTT Worker
```bash
cd backend_api
python -m app.workers.mqtt_worker
```

#### Terminal 3: Dashboard (React)
```bash
cd dashboard
npm install
npm run dev
```

**Truy cập**: http://localhost:3000

#### Terminal 4: Edge App
```bash
cd ai_edge_app
python main.py
```

## Bước 6: Kiểm tra

1. **Health Check**: http://localhost:8000/health
2. **API Docs**: http://localhost:8000/docs
3. **Dashboard**: http://localhost:8501

## Troubleshooting

### Lỗi kết nối Database
- Kiểm tra PostgreSQL đang chạy
- Kiểm tra DATABASE_URL trong .env
- Kiểm tra quyền truy cập database

### Lỗi MQTT
- Kiểm tra Mosquitto đang chạy: `mosquitto -v`
- Test kết nối: `mosquitto_sub -t test`

### Lỗi Camera
- Kiểm tra camera có sẵn: `ls /dev/video*` (Linux) hoặc Device Manager (Windows)
- Thay đổi camera source trong config

### Model files không tìm thấy
- Đảm bảo model files đã được đặt trong `ai_edge_app/models/`
- Kiểm tra tên file chính xác trong code

## Development Tips

1. **Hot Reload**: Backend và Dashboard hỗ trợ auto-reload khi code thay đổi
2. **Logs**: Xem logs trong `ai_edge_app/logs/` hoặc console output
3. **Database Migration**: Sử dụng Alembic cho migration (cần setup thêm)
4. **Testing**: Thêm unit tests trong từng module

## Next Steps

1. Train models với datasets (UTKFace, FER2013, FairFace)
2. Convert models sang ONNX format
3. Tối ưu hóa models (quantization, pruning)
4. Thêm authentication cho production
5. Setup monitoring và logging
6. Deploy lên cloud (AWS, GCP, Azure)

