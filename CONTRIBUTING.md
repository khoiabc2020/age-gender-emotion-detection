# 🤝 CONTRIBUTING GUIDE

**Smart Retail AI - Hướng dẫn đóng góp cho dự án**

Cảm ơn bạn đã quan tâm đến việc đóng góp cho dự án Smart Retail AI! 🎉

---

## 📋 MỤC LỤC

1. [Code of Conduct](#code-of-conduct)
2. [Cách đóng góp](#cách-đóng-góp)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Pull Request Process](#pull-request-process)
7. [Git Workflow](#git-workflow)

---

## 📜 CODE OF CONDUCT

### Nguyên tắc
- ✅ Tôn trọng mọi người
- ✅ Cởi mở với ý kiến khác
- ✅ Tập trung vào vấn đề, không công kích cá nhân
- ✅ Giúp đỡ người mới

### Không chấp nhận
- ❌ Ngôn ngữ xúc phạm hoặc phân biệt đối xử
- ❌ Quấy rối hoặc trolling
- ❌ Spam hoặc quảng cáo
- ❌ Chia sẻ thông tin cá nhân của người khác

---

## 🎯 CÁCH ĐÓNG GÓP

### 1. Báo cáo Bug 🐛

**Trước khi báo cáo:**
- Kiểm tra [Issues](https://github.com/your-org/smart-retail-ai/issues) xem bug đã được báo cáo chưa
- Đảm bảo bạn đang dùng phiên bản mới nhất

**Thông tin cần cung cấp:**
- Mô tả bug rõ ràng
- Các bước để tái hiện
- Kết quả mong đợi vs thực tế
- Screenshots (nếu có)
- Môi trường (OS, Python version, Node version)

**Template:**
```markdown
## Bug Description
[Mô tả ngắn gọn]

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
[Kết quả mong đợi]

## Actual Behavior
[Kết quả thực tế]

## Environment
- OS: Windows 10 / macOS / Linux
- Python: 3.11
- Node: 18.x
```

### 2. Đề xuất Feature ✨

**Trước khi đề xuất:**
- Kiểm tra [Roadmap](docs/ROADMAP.md) xem feature đã có trong kế hoạch chưa
- Tìm kiếm Issues xem có ai đề xuất tương tự chưa

**Thông tin cần cung cấp:**
- Mô tả feature
- Tại sao feature này hữu ích
- Cách implement (nếu có ý tưởng)
- Mockups/designs (nếu có)

### 3. Đóng góp Code 💻

**Các loại contribution:**
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Tests

**Quy trình:**
1. Fork repository
2. Tạo branch mới
3. Implement changes
4. Write tests
5. Update documentation
6. Submit Pull Request

---

## 🛠️ DEVELOPMENT SETUP

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git
- Docker (optional)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/smart-retail-ai.git
cd smart-retail-ai
```

### 2. Backend Setup
```bash
cd backend_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black flake8 mypy

# Setup database
# Edit .env file
cp .env.example .env

# Run migrations (if any)
# alembic upgrade head

# Run tests
pytest

# Run server
uvicorn app.main:app --reload
```

### 3. Frontend Setup
```bash
cd dashboard

# Install dependencies
npm install

# Install dev dependencies (already in package.json)

# Setup environment
cp .env.example .env.local

# Run tests
npm test

# Run dev server
npm run dev
```

### 4. Edge App Setup
```bash
cd ai_edge_app

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
python main.py
```

---

## 📝 CODING STANDARDS

### Python (Backend & Edge App)

#### Style Guide
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [Flake8](https://flake8.pycqa.org/) for linting
- Use type hints

**Example:**
```python
from typing import List, Optional

def get_users(
    limit: int = 10,
    offset: int = 0,
    active_only: bool = True
) -> List[User]:
    """
    Get list of users with pagination.
    
    Args:
        limit: Maximum number of users to return
        offset: Number of users to skip
        active_only: Only return active users
        
    Returns:
        List of User objects
    """
    # Implementation
    pass
```

#### Code Organization
```python
# 1. Standard library imports
import os
import sys
from typing import List

# 2. Third-party imports
import numpy as np
from fastapi import FastAPI

# 3. Local imports
from app.models import User
from app.services import UserService
```

#### Naming Conventions
- **Variables**: `snake_case`
- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

### JavaScript/React (Frontend)

#### Style Guide
- Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use [ESLint](https://eslint.org/) for linting
- Use [Prettier](https://prettier.io/) for formatting

**Example:**
```javascript
/**
 * Fetch user data from API
 * @param {number} userId - User ID
 * @returns {Promise<User>} User object
 */
const fetchUser = async (userId) => {
  try {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching user:', error);
    throw error;
  }
};
```

#### Component Structure
```javascript
// 1. Imports
import React, { useState, useEffect } from 'react';
import { Button, Card } from 'antd';
import { fetchUser } from '../services/api';

// 2. Component
const UserCard = ({ userId }) => {
  // 3. State
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // 4. Effects
  useEffect(() => {
    loadUser();
  }, [userId]);

  // 5. Handlers
  const loadUser = async () => {
    setLoading(true);
    try {
      const data = await fetchUser(userId);
      setUser(data);
    } finally {
      setLoading(false);
    }
  };

  // 6. Render
  return (
    <Card loading={loading}>
      {user && <div>{user.name}</div>}
    </Card>
  );
};

// 7. Export
export default UserCard;
```

---

## 🧪 TESTING

### Backend Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_auth.py

# Run with verbose output
pytest -v
```

**Example Test:**
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_user():
    response = client.post(
        "/users/",
        json={"email": "test@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    assert "id" in response.json()
```

### Frontend Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm test -- --watch
```

**Example Test:**
```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import UserCard from './UserCard';

describe('UserCard', () => {
  it('renders user name', () => {
    render(<UserCard userId={1} />);
    expect(screen.getByText('John Doe')).toBeInTheDocument();
  });

  it('handles click event', () => {
    const handleClick = jest.fn();
    render(<UserCard userId={1} onClick={handleClick} />);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalled();
  });
});
```

---

## 🔄 PULL REQUEST PROCESS

### 1. Tạo Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/bug-description
```

### 2. Implement Changes

- Write clean, readable code
- Follow coding standards
- Add tests for new features
- Update documentation
- Commit frequently with clear messages

### 3. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(auth): add JWT authentication"
git commit -m "fix(api): handle null user in get_user endpoint"
git commit -m "docs: update README with new setup instructions"
git commit -m "test(auth): add tests for login endpoint"
```

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request

**PR Template:**
```markdown
## Description
[Mô tả ngắn gọn về thay đổi]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- Change 1
- Change 2
- ...

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally
```

### 6. Code Review

- Respond to feedback promptly
- Make requested changes
- Push updates to same branch
- Request re-review when ready

### 7. Merge

- Squash commits (if many small commits)
- Delete branch after merge

---

## 🌿 GIT WORKFLOW

### Branch Strategy

```
main (production)
  ├── develop (development)
  │   ├── feature/user-auth
  │   ├── feature/dashboard
  │   └── fix/login-bug
  └── hotfix/critical-bug
```

### Workflow

1. **Feature Development**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/my-feature
   # ... work on feature ...
   git push origin feature/my-feature
   # Create PR to develop
   ```

2. **Bug Fixes**
   ```bash
   git checkout develop
   git checkout -b fix/bug-description
   # ... fix bug ...
   git push origin fix/bug-description
   # Create PR to develop
   ```

3. **Hotfixes**
   ```bash
   git checkout main
   git checkout -b hotfix/critical-issue
   # ... fix issue ...
   git push origin hotfix/critical-issue
   # Create PR to main AND develop
   ```

---

## 📚 RESOURCES

### Documentation
- [README.md](README.md) - Project overview
- [docs/](docs/) - Technical documentation
- [HUONG_DAN_HOC_TAP_VA_SU_DUNG.md](HUONG_DAN_HOC_TAP_VA_SU_DUNG.md) - Learning guide

### Tools
- [Black](https://black.readthedocs.io/) - Python formatter
- [Flake8](https://flake8.pycqa.org/) - Python linter
- [ESLint](https://eslint.org/) - JavaScript linter
- [Prettier](https://prettier.io/) - JavaScript formatter

### Learning
- [Python Best Practices](https://docs.python-guide.org/)
- [React Best Practices](https://react.dev/learn)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## ❓ QUESTIONS?

- 📧 Email: your-email@example.com
- 💬 Discord: [Join our server](https://discord.gg/your-server)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/smart-retail-ai/issues)

---

**Cảm ơn bạn đã đóng góp!** 🙏

**Happy Coding!** 🚀
