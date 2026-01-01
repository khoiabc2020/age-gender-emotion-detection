# 🧹 CODE CLEANUP PLAN - PROFESSIONAL REFACTORING

**Goal**: Make codebase professional for recruiters  
**Date**: 2026-01-02

---

## 🎯 OBJECTIVES

1. ✅ Remove AI-generated comments
2. ✅ Standardize code style
3. ✅ Add professional docstrings
4. ✅ Remove debug code
5. ✅ Improve naming conventions
6. ✅ Add proper error handling
7. ✅ Optimize imports
8. ✅ Add type hints

---

## 📋 CLEANUP CHECKLIST

### Phase 1: Remove AI Markers
- [ ] Remove "Tuần X" comments
- [ ] Remove Vietnamese debug comments
- [ ] Remove excessive emoji in comments
- [ ] Remove "AI-generated" markers
- [ ] Standardize to English comments

### Phase 2: Code Quality
- [ ] Add proper type hints (Python)
- [ ] Add JSDoc comments (JavaScript)
- [ ] Standardize naming (snake_case Python, camelCase JS)
- [ ] Remove unused imports
- [ ] Remove dead code
- [ ] Fix linting errors

### Phase 3: Documentation
- [ ] Professional docstrings (Google style)
- [ ] Clear function descriptions
- [ ] Parameter documentation
- [ ] Return value documentation
- [ ] Usage examples

### Phase 4: Structure
- [ ] Organize imports (standard, third-party, local)
- [ ] Group related functions
- [ ] Separate concerns properly
- [ ] Add __all__ exports

---

## 🔧 TOOLS TO USE

### Python
```bash
# Format code
black .

# Sort imports
isort .

# Lint
pylint src/
flake8 src/

# Type check
mypy src/
```

### JavaScript
```bash
# Format
npm run format

# Lint
npm run lint

# Fix auto-fixable issues
npm run lint:fix
```

---

## 📂 FILES TO CLEAN

### Priority 1: Core Files
- [ ] `ai_edge_app/main.py`
- [ ] `backend_api/app/main.py`
- [ ] `dashboard/src/App.jsx`

### Priority 2: Important Modules
- [ ] `ai_edge_app/src/classifiers/multitask_classifier.py`
- [ ] `ai_edge_app/src/detectors/*.py`
- [ ] `backend_api/app/api/*.py`
- [ ] `dashboard/src/pages/*.jsx`

### Priority 3: Utilities
- [ ] `ai_edge_app/src/utils/*.py`
- [ ] `backend_api/app/services/*.py`
- [ ] `dashboard/src/services/*.js`

---

## ✨ CLEANUP EXAMPLES

### Before (AI-generated style):
```python
"""
Face Detection Module
Tuần 3: Nhận diện khuôn mặt với RetinaFace
TODO: Cải thiện accuracy
"""

def detect_faces(img):
    # Detect faces trong ảnh
    faces = model.detect(img)  # TODO: optimize này
    return faces  # trả về list faces 😊
```

### After (Professional style):
```python
"""Face detection using RetinaFace model.

This module provides face detection functionality optimized for
real-time edge computing applications.
"""

from typing import List, Tuple
import numpy as np

def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in the input image.
    
    Args:
        image: Input image in BGR format (H, W, 3)
        
    Returns:
        List of bounding boxes as (x, y, w, h) tuples
        
    Example:
        >>> img = cv2.imread('test.jpg')
        >>> faces = detect_faces(img)
        >>> print(f"Found {len(faces)} faces")
    """
    faces = model.detect(image)
    return faces
```

---

## 🚀 IMPLEMENTATION STEPS

### Step 1: Backup
```bash
git checkout -b code-cleanup
git add -A
git commit -m "Backup before cleanup"
```

### Step 2: Run Auto-formatters
```bash
# Python
cd ai_edge_app && black . && isort .
cd backend_api && black . && isort .

# JavaScript
cd dashboard && npm run format
```

### Step 3: Manual Review
- Review each file
- Remove AI markers
- Improve docstrings
- Add type hints

### Step 4: Test
```bash
# Python
pytest

# JavaScript
npm test
```

### Step 5: Commit
```bash
git add -A
git commit -m "refactor: Professional code cleanup"
git push origin code-cleanup
```

---

## 📊 EXPECTED IMPROVEMENTS

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Type Coverage | ~30% | ~90% | 85%+ |
| Docstring Coverage | ~50% | ~95% | 90%+ |
| Lint Score | 7/10 | 9.5/10 | 9+ |
| AI Markers | Many | None | 0 |

---

## 🎯 PROFESSIONAL STANDARDS

### Python (PEP 8 + Google Style)
```python
"""Module docstring.

Detailed description of what this module does.
"""

from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class ProfessionalClass:
    """Short description.
    
    Detailed description of the class purpose and usage.
    
    Attributes:
        attribute_name: Description of attribute
        
    Example:
        >>> obj = ProfessionalClass()
        >>> result = obj.method()
    """
    
    def __init__(self, param: str) -> None:
        """Initialize the class.
        
        Args:
            param: Description of parameter
        """
        self.param = param
    
    def method(self, arg: int) -> Optional[str]:
        """Short description of method.
        
        Args:
            arg: Description of argument
            
        Returns:
            Description of return value, or None if not found
            
        Raises:
            ValueError: If arg is negative
        """
        if arg < 0:
            raise ValueError("arg must be non-negative")
        return str(arg)
```

### JavaScript (JSDoc + Airbnb Style)
```javascript
/**
 * Professional React component
 * @component
 */

import React from 'react';
import PropTypes from 'prop-types';

/**
 * Dashboard component for analytics display
 * 
 * @param {Object} props - Component props
 * @param {Array} props.data - Analytics data
 * @param {Function} props.onUpdate - Update callback
 * @returns {React.Element} Dashboard component
 */
const Dashboard = ({ data, onUpdate }) => {
  return (
    <div className="dashboard">
      {/* Component content */}
    </div>
  );
};

Dashboard.propTypes = {
  data: PropTypes.array.isRequired,
  onUpdate: PropTypes.func,
};

Dashboard.defaultProps = {
  onUpdate: () => {},
};

export default Dashboard;
```

---

## ✅ COMPLETION CRITERIA

- [ ] No "Tuần X" comments
- [ ] No Vietnamese debug comments
- [ ] No emoji in code comments
- [ ] All functions have docstrings
- [ ] Type hints on 90%+ functions
- [ ] Pylint score > 9.0
- [ ] ESLint 0 errors
- [ ] All tests passing

---

**Ready to start cleanup!** 🚀
