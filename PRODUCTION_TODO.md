# ✅ PRODUCTION TODO - SMART RETAIL AI

**Timeline**: 2-4 tuần  
**Last Updated**: 2025-12-31  
**Status**: 📋 Ready to Start

---

## 🎯 MỤC TIÊU

Đưa Smart Retail AI từ development lên production với:
- ✅ High availability & scalability
- ✅ Security hardened
- ✅ Full monitoring & logging
- ✅ Performance optimized
- ✅ Production SLA

---

## 📋 CHECKLIST TỔNG QUAN

### ✅ ĐÃ HOÀN THÀNH
- [x] Backend API hoàn chỉnh
- [x] Frontend Dashboard (6 pages)
- [x] Edge AI App
- [x] AI Agent integration
- [x] Model training pipeline
- [x] Docker configuration
- [x] Basic documentation
- [x] Git repository

### ⏳ CẦN HOÀN THÀNH (12 Tasks)
- [x] **1. Full Model Training** [OK] COMPLETED (8 giờ) - QUAN TRỌNG NHẤT
- [ ] **2. Testing & QA** (3-4 ngày)
- [ ] **3. Security Hardening** (2-3 ngày)
- [ ] **4. Environment Setup** (1 ngày)
- [ ] **5. Database Migration** (1 ngày)
- [ ] **6. CI/CD Pipeline** (2 ngày)
- [ ] **7. Monitoring Setup** (2 ngày)
- [ ] **8. Performance Optimization** (2 ngày)
- [ ] **9. SSL/HTTPS Setup** (1 ngày)
- [ ] **10. Backup Strategy** (1 ngày)
- [ ] **11. Staging Deployment** (1 ngày)
- [ ] **12. Production Deployment** (1 ngày)

**Total**: ~14-16 ngày làm việc (2-3 tuần)

---

## 🚀 PHASE 1: MODEL TRAINING (Day 1)

### ⭐ Task 1: Full Model Training
**Priority**: 🔴 CRITICAL  
**Time**: 6-8 giờ (GPU) hoặc chạy overnight  
**Status**: ⏳ Chưa làm

#### Action Items:
```bash
# 1. Chuẩn bị datasets
cd training_experiments
# Datasets đã có từ Kaggle cache

# 2. Chạy full training (khuyến nghị)
python train_10x_automated.py

# Hoặc training đơn
python train_week2_lightweight.py --epochs 50 --batch_size 32
```

#### Expected Results:
- ✅ Gender Accuracy: > 90%
- ✅ Emotion Accuracy: > 75%
- ✅ Age MAE: < 4.0 years
- ✅ Model size: ~25MB
- ✅ ONNX exported

#### Deliverables:
- [ ] Trained model (.pth)
- [ ] ONNX model (.onnx)
- [ ] Training metrics & charts
- [ ] Model evaluation report

---

## 🧪 PHASE 2: TESTING & QA (Days 2-5)

### Task 2: Backend Testing
**Priority**: 🔴 HIGH  
**Time**: 2 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**2.1 Unit Tests** (Day 2 Morning)
```bash
cd backend_api

# Install test dependencies
pip install pytest pytest-cov pytest-asyncio httpx

# Tạo tests cho các modules chính
tests/
├── test_auth.py           # Authentication tests
├── test_analytics.py      # Analytics API tests
├── test_ai_agent.py       # AI Agent tests
├── test_ads.py            # Ads management tests
└── test_database.py       # Database tests

# Run tests
pytest tests/ -v --cov=app --cov-report=html
```

**Target**: Coverage > 80%

**2.2 Integration Tests** (Day 2 Afternoon)
```bash
# Test API endpoints
pytest tests/integration/ -v

# Test với real database
pytest tests/integration/ --db=postgresql
```

**2.3 Load Testing** (Day 3 Morning)
```bash
# Install locust
pip install locust

# Tạo load test script
# tests/load_test.py

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

**Target**: Handle 1000+ concurrent users

**2.4 Security Testing** (Day 3 Afternoon)
```bash
# Install security tools
pip install bandit safety

# Run security scans
bandit -r app/ -f json -o security_report.json
safety check

# SQL injection tests
sqlmap -u "http://localhost:8000/api/v1/analytics"
```

### Task 3: Frontend Testing
**Priority**: 🟡 MEDIUM  
**Time**: 1 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**3.1 Unit Tests** (Day 4 Morning)
```bash
cd dashboard

# Install test dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom vitest

# Tạo tests cho components
src/
├── __tests__/
│   ├── Dashboard.test.tsx
│   ├── Analytics.test.tsx
│   ├── AIAgent.test.tsx
│   └── AdsManagement.test.tsx

# Run tests
npm test
npm run test:coverage
```

**3.2 E2E Tests** (Day 4 Afternoon)
```bash
# Install Playwright
npm install --save-dev @playwright/test

# Tạo E2E tests
e2e/
├── login.spec.ts
├── dashboard.spec.ts
└── ai-agent.spec.ts

# Run E2E tests
npx playwright test
```

**3.3 Performance Audit** (Day 5 Morning)
```bash
# Build production
npm run build

# Run Lighthouse
npm install -g lighthouse
lighthouse http://localhost:3000 --output=html --output-path=./lighthouse-report.html
```

**Target**: Lighthouse Score > 90

### Task 4: Edge App Testing
**Priority**: 🟡 MEDIUM  
**Time**: 0.5 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:
```bash
cd ai_edge_app

# Unit tests
pytest tests/ -v

# Memory leak testing
python -m memory_profiler main.py

# Performance profiling
python -m cProfile -o profile.stats main.py
```

---

## 🔒 PHASE 3: SECURITY HARDENING (Days 6-7)

### Task 5: Environment & Secrets
**Priority**: 🔴 CRITICAL  
**Time**: 1 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**5.1 Environment Files** (Day 6 Morning)
```bash
# Tạo environment files cho các môi trường
backend_api/
├── .env.development      # Development
├── .env.staging          # Staging
├── .env.production       # Production (không commit)
└── .env.example          # Template

dashboard/
├── .env.development
├── .env.staging
├── .env.production
└── .env.example
```

**5.2 Secrets Management** (Day 6 Afternoon)
```bash
# Option 1: Docker Secrets
docker secret create db_password db_password.txt
docker secret create jwt_secret jwt_secret.txt

# Option 2: AWS Secrets Manager
aws secretsmanager create-secret --name retail-ai/db-password --secret-string "xxx"
aws secretsmanager create-secret --name retail-ai/jwt-secret --secret-string "xxx"

# Option 3: HashiCorp Vault
vault kv put secret/retail-ai db_password="xxx" jwt_secret="xxx"
```

**5.3 Security Checklist**
- [ ] Change default passwords (`admin`/`admin123`)
- [ ] Generate strong JWT secret (64+ characters)
- [ ] Generate strong database password
- [ ] Rotate API keys (Google AI, OpenAI)
- [ ] Setup API rate limiting
- [ ] Enable CORS properly
- [ ] Setup CSP headers
- [ ] Enable SQL injection prevention
- [ ] Enable XSS protection

### Task 6: SSL/HTTPS Setup
**Priority**: 🔴 HIGH  
**Time**: 0.5 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**6.1 Get SSL Certificate** (Day 7 Morning)
```bash
# Option 1: Let's Encrypt (Free)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Option 2: Buy from CA
# - Namecheap, GoDaddy, etc.

# Option 3: CloudFlare (Free)
# - Enable CloudFlare proxy
```

**6.2 Configure Nginx** (Day 7 Morning)
```nginx
# /etc/nginx/sites-available/retail-ai
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Backend proxy
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🗄️ PHASE 4: DATABASE & INFRASTRUCTURE (Days 8-9)

### Task 7: Database Setup
**Priority**: 🔴 HIGH  
**Time**: 1 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**7.1 Production Database** (Day 8 Morning)
```bash
# Option 1: Self-hosted PostgreSQL
sudo apt install postgresql-14
sudo -u postgres createdb retail_analytics_prod

# Option 2: Managed Database (Recommended)
# - AWS RDS PostgreSQL
# - Digital Ocean Managed Database
# - Google Cloud SQL

# Configure connection
DATABASE_URL=postgresql://user:pass@host:5432/retail_analytics_prod
```

**7.2 Database Migration** (Day 8 Afternoon)
```bash
cd backend_api

# Install Alembic
pip install alembic

# Initialize migrations
alembic init migrations

# Create initial migration
alembic revision --autogenerate -m "initial"

# Run migrations
alembic upgrade head
```

**7.3 Database Optimization** (Day 8 Afternoon)
```sql
-- Add indexes
CREATE INDEX idx_analytics_timestamp ON analytics(timestamp);
CREATE INDEX idx_analytics_customer_id ON analytics(customer_id);
CREATE INDEX idx_ads_category ON ads(category);

-- Enable query optimization
ANALYZE;
VACUUM;
```

### Task 8: Backup Strategy
**Priority**: 🔴 HIGH  
**Time**: 0.5 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**8.1 Automated Backups** (Day 9 Morning)
```bash
# Create backup script
# /usr/local/bin/backup_db.sh

#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgresql"
DB_NAME="retail_analytics_prod"

# Backup database
pg_dump $DB_NAME | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp $BACKUP_DIR/backup_$TIMESTAMP.sql.gz s3://my-backups/postgresql/
```

**8.2 Cron Job**
```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /usr/local/bin/backup_db.sh
```

**8.3 Backup Testing**
- [ ] Test restore from backup monthly
- [ ] Document restore procedure
- [ ] Store backups in multiple locations

---

## 📊 PHASE 5: MONITORING & LOGGING (Days 10-11)

### Task 9: Monitoring Setup
**Priority**: 🟡 MEDIUM  
**Time**: 2 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**9.1 Prometheus + Grafana** (Day 10)
```bash
# Install Prometheus
docker run -d -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Install Grafana
docker run -d -p 3001:3000 grafana/grafana

# Configure Prometheus scraping
# prometheus.yml
scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['localhost:8000']
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

**9.2 Application Metrics** (Day 10)
```python
# backend_api/app/middleware/metrics.py
from prometheus_client import Counter, Histogram

request_count = Counter('http_requests_total', 'Total requests')
request_duration = Histogram('http_request_duration_seconds', 'Request duration')

# Add to FastAPI middleware
```

**9.3 Grafana Dashboards** (Day 11)
- [ ] System metrics (CPU, RAM, Disk)
- [ ] Application metrics (Requests, Latency, Errors)
- [ ] Database metrics (Connections, Queries, Slow queries)
- [ ] Business metrics (Active users, Ad impressions)

**9.4 Alerting** (Day 11)
```yaml
# alertmanager.yml
route:
  receiver: 'email'
  
receivers:
  - name: 'email'
    email_configs:
      - to: 'ops@yourcompany.com'
        from: 'alerts@yourcompany.com'

# Alert rules
groups:
  - name: 'backend'
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status="500"}[5m]) > 0.05
        for: 5m
      - alert: HighLatency
        expr: http_request_duration_seconds > 1
        for: 5m
```

### Task 10: Logging Setup
**Priority**: 🟡 MEDIUM  
**Time**: 1 ngày (included in Task 9)  
**Status**: ⏳ Chưa làm

#### Action Items:

**10.1 Centralized Logging** (Day 11)
```bash
# Option 1: ELK Stack
docker-compose -f docker-compose.elk.yml up -d

# Option 2: Cloud logging
# - AWS CloudWatch
# - Google Cloud Logging
# - DataDog
```

**10.2 Application Logging**
```python
# backend_api/app/core/logging.py
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }
        return json.dumps(log_record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('/var/log/retail-ai/app.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🚀 PHASE 6: CI/CD PIPELINE (Days 12-13)

### Task 11: CI/CD Setup
**Priority**: 🟡 MEDIUM  
**Time**: 2 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**11.1 GitHub Actions** (Day 12)
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          cd backend_api
          pip install -r requirements.txt
      - name: Run tests
        run: |
          cd backend_api
          pytest tests/ --cov=app
      - name: Security scan
        run: |
          pip install bandit safety
          bandit -r backend_api/app/
          safety check

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd dashboard
          npm ci
      - name: Run tests
        run: |
          cd dashboard
          npm test
      - name: Build
        run: |
          cd dashboard
          npm run build

  build-docker:
    needs: [test-backend, test-frontend]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build images
        run: docker-compose build
      - name: Push to registry
        run: |
          docker tag retail-ai-backend:latest yourregistry/retail-ai-backend:${{ github.sha }}
          docker push yourregistry/retail-ai-backend:${{ github.sha }}
```

**11.2 Deployment Pipeline** (Day 13)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PRODUCTION_HOST }}
          username: ${{ secrets.PRODUCTION_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/retail-ai
            git pull
            docker-compose down
            docker-compose up -d
```

---

## 🎯 PHASE 7: PERFORMANCE OPTIMIZATION (Day 14)

### Task 12: Performance Tuning
**Priority**: 🟡 MEDIUM  
**Time**: 1 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**12.1 Backend Optimization** (Day 14 Morning)
```python
# Add caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# Cache responses
from fastapi_cache.decorator import cache

@router.get("/analytics/summary")
@cache(expire=300)  # Cache for 5 minutes
async def get_summary():
    return {"data": "..."}

# Database connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

**12.2 Frontend Optimization** (Day 14 Afternoon)
```typescript
// Code splitting
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Analytics = lazy(() => import('./pages/Analytics'));

// Image optimization
import { Image } from 'antd';

<Image
  src="/images/ad.jpg"
  placeholder={<Spin />}
  loading="lazy"
/>

// Bundle optimization
// vite.config.ts
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom'],
          'antd': ['antd'],
        }
      }
    }
  }
}
```

**12.3 CDN Setup** (Day 14 Afternoon)
```bash
# Option 1: CloudFlare (Free)
# - Add domain to CloudFlare
# - Enable proxy & caching

# Option 2: AWS CloudFront
aws cloudfront create-distribution \
  --origin-domain-name yourdomain.com

# Option 3: Nginx caching
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

---

## 🌐 PHASE 8: STAGING & PRODUCTION DEPLOYMENT (Days 15-16)

### Task 13: Staging Deployment
**Priority**: 🔴 HIGH  
**Time**: 1 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**13.1 Setup Staging Environment** (Day 15)
```bash
# Create staging server (example: DigitalOcean)
# - 4GB RAM, 2 vCPUs
# - Ubuntu 22.04 LTS

# Setup Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Clone repository
git clone https://github.com/your-org/smart-retail-ai.git
cd smart-retail-ai

# Configure environment
cp backend_api/.env.staging backend_api/.env
cp dashboard/.env.staging dashboard/.env.local

# Deploy
docker-compose -f docker-compose.staging.yml up -d
```

**13.2 Staging Testing** (Day 15)
- [ ] Smoke tests (all pages load)
- [ ] Integration tests (API + Frontend)
- [ ] User acceptance testing (UAT)
- [ ] Performance testing
- [ ] Security testing

### Task 14: Production Deployment
**Priority**: 🔴 CRITICAL  
**Time**: 1 ngày  
**Status**: ⏳ Chưa làm

#### Action Items:

**14.1 Pre-deployment Checklist** (Day 16 Morning)
- [ ] All tests passing
- [ ] Security audit completed
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] SSL certificates ready
- [ ] DNS configured
- [ ] Rollback plan documented

**14.2 Deploy to Production** (Day 16 Afternoon)
```bash
# Production server
ssh user@production-server

# Setup
cd /opt/retail-ai
git pull origin main

# Update environment
cp .env.production .env

# Deploy
docker-compose down
docker-compose up -d

# Verify
docker-compose ps
docker-compose logs -f
```

**14.3 Post-deployment** (Day 16 Evening)
- [ ] Verify all services running
- [ ] Test all critical paths
- [ ] Monitor logs for errors
- [ ] Check performance metrics
- [ ] Notify stakeholders

---

## 📝 QUICK REFERENCE

### Priorities
- 🔴 **CRITICAL**: Must have for production
- 🟡 **HIGH**: Important but can defer
- 🟢 **MEDIUM**: Nice to have

### Timeline Summary
| Phase | Days | Status |
|-------|------|--------|
| Model Training | 1 | ⏳ |
| Testing & QA | 4 | ⏳ |
| Security | 2 | ⏳ |
| Infrastructure | 2 | ⏳ |
| Monitoring | 2 | ⏳ |
| CI/CD | 2 | ⏳ |
| Performance | 1 | ⏳ |
| Deployment | 2 | ⏳ |
| **TOTAL** | **16 days** | |

### Costs Estimate

**Option 1: Cloud (AWS/GCP)**
- EC2/Compute (t3.medium x2): $60/month
- RDS PostgreSQL: $30/month
- S3/Storage: $10/month
- CloudWatch/Monitoring: $20/month
- **Total**: ~$120/month

**Option 2: VPS (DigitalOcean/Linode)**
- Droplet (4GB x2): $48/month
- Managed Database: $15/month
- Spaces/Storage: $5/month
- **Total**: ~$68/month

**Option 3: Self-Hosted**
- Server hardware: One-time cost
- Internet: $50/month
- UPS/Backup: One-time cost
- **Total**: ~$50/month + hardware

---

## ✅ CRITICAL PATH (Minimum Viable Production)

Nếu cần deploy nhanh (1 tuần):

### Day 1: Model & Core
- [x] Full model training (overnight)
- [ ] Change default passwords
- [ ] Setup production .env

### Day 2-3: Testing
- [ ] Backend API tests (critical paths)
- [ ] Frontend smoke tests
- [ ] Security scan

### Day 4: Infrastructure
- [ ] Production server setup
- [ ] Database setup & migration
- [ ] SSL certificate

### Day 5: Deploy
- [ ] Staging deployment & testing
- [ ] Production deployment
- [ ] Monitoring setup (basic)

### Day 6-7: Monitor & Fix
- [ ] Monitor logs & metrics
- [ ] Fix critical issues
- [ ] Performance tuning

---

## 🆘 SUPPORT RESOURCES

### Documentation
- **Full guide**: [PRODUCTION_ROADMAP.md](docs/PRODUCTION_ROADMAP.md)
- **Security**: [SECURITY.md](docs/SECURITY.md)
- **Setup**: [SETUP.md](docs/SETUP.md)
- **CI/CD**: [CI_CD.md](docs/CI_CD.md)

### Tools
- **Testing**: pytest, locust, playwright
- **Security**: bandit, safety, sqlmap
- **Monitoring**: Prometheus, Grafana, Sentry
- **CI/CD**: GitHub Actions, Docker

### Contacts
- Technical Lead: [Your Name]
- DevOps: [Name]
- Security: [Name]

---

**🎯 START HERE**: Task 1 - Full Model Training

**📚 READ**: [PRODUCTION_READY.md](PRODUCTION_READY.md) for complete overview

**Last Updated**: 2025-12-31
