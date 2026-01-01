# Backend API

FastAPI-based REST API for Smart Retail AI system.

---

## 🚀 **Features**

- **REST API**: Complete RESTful endpoints
- **WebSocket**: Real-time communication
- **Authentication**: JWT-based security
- **Analytics**: Customer interaction tracking
- **AI Agent**: Google AI & ChatGPT integration
- **MQTT**: Edge device communication
- **Database**: PostgreSQL with SQLAlchemy
- **Background Workers**: Async task processing

---

## 📁 **Structure**

```
backend_api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/                 # API routes
│   │   ├── analytics.py     # Analytics endpoints
│   │   ├── auth.py          # Authentication
│   │   └── ai_agent.py      # AI Agent endpoints
│   ├── core/                # Core functionality
│   │   ├── config.py        # Configuration
│   │   └── security.py      # Security utilities
│   ├── db/                  # Database
│   │   ├── database.py      # DB connection
│   │   └── models.py        # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   └── workers/             # Background workers
├── tests/                   # Unit tests
├── requirements.txt         # Dependencies
└── Dockerfile              # Container config
```

---

## 🛠️ **Setup**

### **Prerequisites:**
- Python 3.9+
- PostgreSQL 13+
- Redis (optional, for caching)

### **Installation:**

```bash
cd backend_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp ../.env.example .env
# Edit .env with your values
```

### **Database Setup:**

```bash
# Create database
createdb smartretail_db

# Run migrations (if using Alembic)
alembic upgrade head
```

---

## 🚀 **Run**

### **Development:**

```bash
# With auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **Production:**

```bash
# With Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### **Docker:**

```bash
docker build -t smartretail-backend .
docker run -p 8000:8000 smartretail-backend
```

---

## 📡 **API Endpoints**

### **Base URL:** `http://localhost:8000`

### **Authentication:**

```http
POST /api/v1/auth/login
POST /api/v1/auth/register
GET  /api/v1/auth/me
```

### **Analytics:**

```http
POST /api/v1/analytics/interactions     # Create interaction
GET  /api/v1/analytics/stats             # Get statistics
GET  /api/v1/analytics/age-by-hour       # Age distribution
GET  /api/v1/analytics/emotion-distribution  # Emotion stats
GET  /api/v1/analytics/gender-distribution   # Gender stats
```

### **AI Agent:**

```http
POST /api/v1/ai/chat                # Chat with AI
POST /api/v1/ai/analyze             # Analyze data
POST /api/v1/ai/generate-report     # Generate report
GET  /api/v1/ai/status              # Check AI status
```

### **WebSocket:**

```
ws://localhost:8000/ws
```

---

## 📖 **API Documentation**

**Swagger UI:** http://localhost:8000/docs  
**ReDoc:** http://localhost:8000/redoc

---

## 🧪 **Testing**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_auth.py
```

---

## 🔒 **Security**

- **JWT Authentication**: Secure token-based auth
- **Password Hashing**: bcrypt
- **CORS**: Configurable origins
- **Input Validation**: Pydantic schemas
- **SQL Injection Prevention**: SQLAlchemy ORM
- **Rate Limiting**: Redis-based (optional)

---

## 🔧 **Configuration**

### **Environment Variables:**

See `../.env.example` for all available options.

**Key variables:**
```env
DATABASE_URL=postgresql://user:pass@localhost/smartretail_db
SECRET_KEY=your-secret-key
GOOGLE_AI_API_KEY=your-google-key
OPENAI_API_KEY=your-openai-key
```

---

## 📊 **Monitoring**

### **Health Check:**

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected"
}
```

---

## 🐛 **Troubleshooting**

### **Database Connection Error:**

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check DATABASE_URL in .env
echo $DATABASE_URL
```

### **Import Errors:**

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### **CORS Issues:**

```python
# In app/main.py, check CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📝 **Development**

### **Code Style:**

```bash
# Format code
black app/

# Check linting
flake8 app/

# Type checking
mypy app/
```

### **Database Migrations:**

```bash
# Create migration
alembic revision -m "description"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 🚀 **Deployment**

### **Docker:**

```bash
docker-compose up -d backend
```

### **Kubernetes:**

```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml
```

---

## 📚 **Documentation**

- **API Docs**: `/docs` (Swagger)
- **Architecture**: `../docs/PROJECT_DETAILS.md`
- **Security**: `../docs/SECURITY.md`
- **Production**: `../docs/PRODUCTION_ROADMAP.md`

---

## 🤝 **Contributing**

See `../CONTRIBUTING.md` for contribution guidelines.

---

## 📄 **License**

MIT License - See `../LICENSE` for details

---

**Version:** 2.0.0  
**Last Updated:** January 2, 2026  
**Status:** Production Ready
