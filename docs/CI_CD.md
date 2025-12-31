# 🚀 CI/CD PIPELINE DOCUMENTATION

**Smart Retail AI - Continuous Integration & Continuous Deployment**

---

## 📋 TỔNG QUAN

Dự án đã được tích hợp CI/CD pipeline hoàn chỉnh sử dụng **GitHub Actions**, tự động hóa:
- ✅ Testing (Backend & Frontend)
- ✅ Code Quality Checks (Linting, Formatting)
- ✅ Docker Image Building
- ✅ Security Scanning
- ✅ Automated Deployment
- ✅ Model Training Pipeline

---

## 🔧 CI/CD WORKFLOWS

### 1. **CI Pipeline** (`.github/workflows/ci.yml`)

Chạy tự động khi:
- Push code lên `main` hoặc `develop`
- Tạo Pull Request

**Jobs bao gồm:**

#### Backend Tests & Linting
- ✅ Python code formatting check (Black)
- ✅ Linting (Flake8)
- ✅ Unit tests (Pytest)
- ✅ Code coverage
- ✅ PostgreSQL service cho testing

#### Frontend Tests & Linting
- ✅ ESLint code quality check
- ✅ React component tests
- ✅ Build verification
- ✅ Code coverage

#### Docker Build Test
- ✅ Build backend Docker image
- ✅ Build frontend Docker image
- ✅ Build edge app Docker image
- ✅ Cache optimization

#### Security Scan
- ✅ Trivy vulnerability scanner
- ✅ Upload results to GitHub Security

---

### 2. **CD Pipeline** (`.github/workflows/cd.yml`)

Chạy khi:
- Push lên `main` branch
- Tạo tag version (v*)
- Manual trigger với environment selection

**Jobs bao gồm:**

#### Build & Push Docker Images
- ✅ Build và push images lên GitHub Container Registry
- ✅ Tag images với version, branch, SHA
- ✅ Multi-service support (backend, frontend, edge)

#### Deploy to Staging
- ✅ Tự động deploy khi push lên `main`
- ✅ Environment: `staging`

#### Deploy to Production
- ✅ Deploy khi tạo tag version
- ✅ Manual trigger với `production` environment
- ✅ Tự động tạo GitHub Release

---

### 3. **Training Pipeline** (`.github/workflows/training.yml`)

Chạy khi:
- Manual trigger với parameters
- Scheduled: Mỗi Chủ nhật 2 AM UTC

**Jobs bao gồm:**

#### Train Model
- ✅ Check datasets
- ✅ Run training với configurable epochs/batch_size
- ✅ Upload training results as artifacts
- ✅ Convert model to ONNX
- ✅ Upload ONNX model

---

## 🛠️ SETUP CI/CD

### 1. **GitHub Repository Setup**

```bash
# Đảm bảo repository có các secrets (nếu cần):
# - GITHUB_TOKEN (tự động có)
# - DOCKER_REGISTRY_TOKEN (nếu dùng registry khác)
```

### 2. **Enable GitHub Actions**

1. Vào repository Settings → Actions → General
2. Enable "Allow all actions and reusable workflows"
3. Save changes

### 3. **Test CI Pipeline**

```bash
# Tạo branch mới
git checkout -b feature/test-ci

# Push code
git push origin feature/test-ci

# Tạo Pull Request
# CI sẽ tự động chạy
```

### 4. **Test CD Pipeline**

```bash
# Tạo tag version
git tag v1.0.0
git push origin v1.0.0

# Hoặc push lên main
git checkout main
git push origin main
```

---

## 📝 TEST FILES

### Backend Tests

**Location:** `backend_api/tests/`

- `test_main.py` - Tests cho FastAPI app
- `test_auth.py` - Tests cho authentication

**Chạy tests:**
```bash
cd backend_api
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html
```

### Frontend Tests

**Location:** `dashboard/src/__tests__/`

- `App.test.jsx` - Basic smoke test

**Chạy tests:**
```bash
cd dashboard
npm test
npm run test:coverage
```

---

## 🔍 CODE QUALITY TOOLS

### Backend

#### Black (Code Formatter)
```bash
cd backend_api
black app/ --check  # Check only
black app/          # Format code
```

#### Flake8 (Linter)
```bash
cd backend_api
flake8 app/
```

#### Pytest (Testing)
```bash
cd backend_api
pytest tests/ -v
```

### Frontend

#### ESLint
```bash
cd dashboard
npm run lint
```

#### Vitest (Testing)
```bash
cd dashboard
npm test
```

---

## 🐳 DOCKER IMAGES

### Build Images Locally

```bash
# Backend
docker build -t smart-retail-api:latest ./backend_api

# Frontend
docker build -t smart-retail-dashboard:latest ./dashboard

# Edge App
docker build -t smart-retail-edge:latest ./ai_edge_app
```

### Pull Images from Registry

```bash
# After CD pipeline runs
docker pull ghcr.io/<username>/smart-retail-api:latest
docker pull ghcr.io/<username>/smart-retail-dashboard:latest
docker pull ghcr.io/<username>/smart-retail-edge:latest
```

---

## 🚀 DEPLOYMENT

### Staging Deployment

Tự động deploy khi push lên `main` branch.

### Production Deployment

**Cách 1: Tạo Tag**
```bash
git tag v1.0.0
git push origin v1.0.0
```

**Cách 2: Manual Trigger**
1. Vào GitHub Actions
2. Chọn "CD Pipeline"
3. Click "Run workflow"
4. Chọn environment: `production`
5. Click "Run workflow"

---

## 📊 MONITORING CI/CD

### View Workflow Runs

1. Vào repository trên GitHub
2. Click tab "Actions"
3. Xem workflow runs và logs

### View Test Results

- **Backend:** Coverage report trong workflow logs
- **Frontend:** Coverage report trong workflow logs
- **Security:** GitHub Security tab

### View Artifacts

1. Vào workflow run
2. Scroll xuống "Artifacts"
3. Download training results, models, etc.

---

## 🔐 SECURITY

### Secrets Management

GitHub Secrets được sử dụng cho:
- `GITHUB_TOKEN` - Tự động có, dùng cho registry login
- Custom secrets có thể thêm trong Settings → Secrets

### Security Scanning

- **Trivy** tự động scan code và dependencies
- Results được upload lên GitHub Security tab
- Fix vulnerabilities được recommend

---

## 🎯 BEST PRACTICES

### 1. **Commit Messages**
```
feat: Add new feature
fix: Fix bug
test: Add tests
ci: Update CI/CD
docs: Update documentation
```

### 2. **Branch Strategy**
- `main` - Production code
- `develop` - Development code
- `feature/*` - Feature branches
- `fix/*` - Bug fixes

### 3. **Pull Requests**
- Tạo PR từ feature branch
- CI sẽ tự động chạy
- Đảm bảo tất cả tests pass trước khi merge

### 4. **Versioning**
- Sử dụng semantic versioning: `v1.0.0`
- Tag releases trên GitHub
- CD pipeline tự động deploy

---

## 🐛 TROUBLESHOOTING

### CI Fails

1. **Check workflow logs:**
   - Vào Actions tab
   - Click vào failed workflow
   - Xem logs để tìm lỗi

2. **Common issues:**
   - Tests fail → Fix tests
   - Linting errors → Run linter locally
   - Build errors → Check Dockerfiles
   - Missing dependencies → Update requirements.txt

### CD Fails

1. **Check deployment logs**
2. **Verify secrets are set**
3. **Check registry permissions**

### Training Pipeline Fails

1. **Check datasets exist**
2. **Verify GPU/resources (if needed)**
3. **Check training logs**

---

## 📚 TÀI LIỆU THAM KHẢO

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)

---

## ✅ CHECKLIST

- [x] CI Pipeline setup
- [x] CD Pipeline setup
- [x] Training Pipeline setup
- [x] Backend tests
- [x] Frontend tests
- [x] Code quality tools
- [x] Docker builds
- [x] Security scanning
- [x] Documentation

---

**Status**: ✅ CI/CD Pipeline Hoàn Chỉnh  
**Last Updated**: 2025-12-30

