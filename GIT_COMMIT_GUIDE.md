# 📋 HƯỚNG DẪN GIT COMMIT - SMART RETAIL AI

**Hướng dẫn chi tiết về những file nên commit và không nên commit**

---

## ✅ FILES NÊN COMMIT (SHOULD COMMIT)

### 📁 **Source Code**
```
✅ Tất cả file .py (Python source code)
✅ Tất cả file .js, .jsx, .ts, .tsx (JavaScript/TypeScript)
✅ Tất cả file .json (config files, package.json, etc.)
✅ Tất cả file .md (documentation)
✅ Tất cả file .yml, .yaml (config files)
✅ Tất cả file .txt (requirements.txt, etc.)
✅ Tất cả file .sh, .bat (scripts)
```

### 📁 **Configuration Files**
```
✅ .gitignore
✅ .dockerignore
✅ docker-compose.yml
✅ Dockerfile
✅ requirements.txt
✅ package.json
✅ package-lock.json (hoặc yarn.lock)
✅ vite.config.js
✅ tailwind.config.js
✅ postcss.config.js
✅ pytest.ini
✅ .flake8
✅ pyproject.toml
✅ .eslintrc.js
✅ vitest.config.js
✅ nginx.conf
```

### 📁 **Documentation**
```
✅ README.md
✅ *.md (tất cả markdown files)
✅ docs/**/*.md
```

### 📁 **CI/CD**
```
✅ .github/workflows/*.yml
✅ .github/workflows/*.yaml
```

### 📁 **Test Files**
```
✅ tests/**/*.py
✅ **/__tests__/**/*.jsx
✅ **/__tests__/**/*.js
```

### 📁 **Config Examples**
```
✅ .env.example
✅ *.env.example
✅ configs/*.json (config templates)
```

### 📁 **Database Schemas**
```
✅ database/init.sql
✅ migrations/*.py
✅ alembic.ini
```

---

## ❌ FILES KHÔNG NÊN COMMIT (SHOULD NOT COMMIT)

### 🔒 **Environment & Secrets**
```
❌ .env
❌ .env.local
❌ .env.production
❌ .env.development
❌ *.env (trừ .env.example)
❌ *.key
❌ *.pem
❌ *.cert
❌ secrets/
```

### 🐍 **Python**
```
❌ __pycache__/
❌ *.pyc
❌ *.pyo
❌ *.pyd
❌ venv/
❌ venv_gpu/
❌ env/
❌ .venv/
❌ *.egg-info/
❌ dist/
❌ build/
```

### 📦 **Node.js**
```
❌ node_modules/
❌ .npm/
❌ .yarn/
❌ dist/
❌ build/
❌ .next/
❌ out/
```

### 🤖 **AI Models (Large Files)**
```
❌ **/models/*.onnx
❌ **/models/*.pt
❌ **/models/*.pth
❌ **/models/*.pkl
❌ **/models/*.h5
❌ **/models/*.ckpt
```

**Lý do:** Model files quá lớn (hàng trăm MB đến GB)
**Giải pháp:** 
- Dùng Git LFS (Git Large File Storage)
- Hoặc lưu trên cloud storage (S3, Google Drive, etc.)
- Hoặc dùng model registry (MLflow, DVC)

### 📊 **Training Data**
```
❌ data/raw/
❌ data/processed/
❌ training_experiments/data/raw/
❌ training_experiments/data/processed/
❌ training_experiments/data/utkface/
❌ training_experiments/data/fer2013/
```

**Lý do:** Datasets quá lớn
**Giải pháp:** 
- Dùng DVC (Data Version Control)
- Hoặc lưu trên cloud storage
- Hoặc dùng dataset registry

### 📝 **Logs**
```
❌ logs/
❌ *.log
❌ *.log.*
❌ npm-debug.log*
❌ yarn-debug.log*
```

### 💾 **Database Files**
```
❌ *.db
❌ *.sqlite
❌ *.sqlite3
❌ *.db-journal
```

### 🗂️ **Checkpoints & Training Results**
```
❌ checkpoints/
❌ training_experiments/checkpoints/
❌ training_experiments/training_results/
❌ *.pth
❌ *.pt
❌ *.ckpt
```

### 🧪 **Test Coverage**
```
❌ .coverage
❌ coverage/
❌ htmlcov/
❌ .pytest_cache/
❌ .nyc_output/
```

### 🐳 **Docker**
```
❌ docker-compose.override.yml (local overrides)
```

### 💻 **IDE & Editor**
```
❌ .vscode/
❌ .idea/
❌ *.swp
❌ *.swo
❌ *.sublime-project
❌ *.sublime-workspace
```

### 🖥️ **OS Files**
```
❌ .DS_Store
❌ Thumbs.db
❌ desktop.ini
❌ $RECYCLE.BIN/
```

### 📓 **Jupyter Notebooks**
```
❌ .ipynb_checkpoints/
❌ *.ipynb (nếu không cần thiết)
```

---

## 🎯 QUY TẮC COMMIT

### 1. **Kiểm tra trước khi commit**
```bash
# Xem những file sẽ được commit
git status

# Xem diff của các file
git diff

# Xem file nào đã được ignore
git status --ignored
```

### 2. **Commit từng phần hợp lý**
```bash
# Commit source code
git add backend_api/app/
git add dashboard/src/
git commit -m "feat: Add new feature"

# Commit config files
git add docker-compose.yml
git add .github/workflows/
git commit -m "ci: Update CI/CD pipeline"

# Commit documentation
git add *.md
git commit -m "docs: Update documentation"
```

### 3. **Không commit file lớn**
```bash
# Kiểm tra kích thước file
git ls-files | xargs ls -lh | sort -k5 -hr | head -20

# Nếu file > 100MB, nên dùng Git LFS
git lfs track "*.onnx"
git lfs track "*.pth"
```

---

## 📦 GIT LFS CHO MODEL FILES

Nếu cần commit model files, dùng Git LFS:

```bash
# Cài đặt Git LFS
git lfs install

# Track model files
git lfs track "*.onnx"
git lfs track "*.pth"
git lfs track "*.pt"

# Commit .gitattributes
git add .gitattributes
git commit -m "chore: Add Git LFS tracking for model files"
```

---

## 🔍 KIỂM TRA TRƯỚC KHI PUSH

### Checklist trước khi push:

- [ ] Không có file `.env` trong commit
- [ ] Không có `node_modules/` trong commit
- [ ] Không có `venv/` hoặc `__pycache__/` trong commit
- [ ] Không có model files lớn (trừ khi dùng Git LFS)
- [ ] Không có logs hoặc database files
- [ ] Không có secrets hoặc API keys
- [ ] Đã test code trước khi commit
- [ ] Commit message rõ ràng và mô tả đúng

### Command để kiểm tra:
```bash
# Xem tất cả files sẽ được commit
git ls-files

# Kiểm tra file lớn
git ls-files | xargs ls -lh | awk '$5 > 10485760 {print $5, $9}'

# Kiểm tra secrets (nếu có script)
python check_api_keys.py
```

---

## 🚨 LỖI THƯỜNG GẶP

### 1. **Commit nhầm file .env**
```bash
# Xóa file khỏi commit (nhưng giữ file local)
git rm --cached .env

# Thêm vào .gitignore
echo ".env" >> .gitignore

# Commit lại
git add .gitignore
git commit -m "fix: Remove .env from git"
```

### 2. **Commit nhầm file lớn**
```bash
# Xóa file khỏi git history (cẩn thận!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/large/file" \
  --prune-empty --tag-name-filter cat -- --all

# Hoặc dùng git-filter-repo (khuyên dùng)
git filter-repo --path path/to/large/file --invert-paths
```

### 3. **Commit nhầm node_modules/**
```bash
# Xóa khỏi git
git rm -r --cached node_modules/

# Đảm bảo có trong .gitignore
echo "node_modules/" >> .gitignore

# Commit
git add .gitignore
git commit -m "fix: Remove node_modules from git"
```

---

## 📚 TÀI LIỆU THAM KHẢO

- [Git Documentation](https://git-scm.com/doc)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [Gitignore Patterns](https://git-scm.com/docs/gitignore)
- [DVC (Data Version Control)](https://dvc.org/)

---

## ✅ SUMMARY

### ✅ **COMMIT:**
- Source code (.py, .js, .jsx, .ts, .tsx)
- Config files (.json, .yml, .yaml, .ini)
- Documentation (.md)
- Scripts (.sh, .bat)
- Test files
- CI/CD workflows
- .env.example files

### ❌ **KHÔNG COMMIT:**
- .env files (secrets)
- node_modules/, venv/
- Model files lớn (.onnx, .pth, .pt)
- Training data lớn
- Logs (*.log)
- Database files (*.db, *.sqlite)
- Checkpoints và training results
- IDE config (.vscode/, .idea/)
- OS files (.DS_Store, Thumbs.db)

---

**Status**: ✅ Complete  
**Last Updated**: 2025-12-30

