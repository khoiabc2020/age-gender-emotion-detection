# 📚 Hướng dẫn Học tập Toàn diện - Smart Retail AI Project

> **Lưu ý**: File này chứa hướng dẫn học tập chi tiết. **KHÔNG XÓA** file này trừ khi được yêu cầu rõ ràng.

## 📋 Mục lục

1. [Tổng quan dự án](#tổng-quan-dự-án)
2. [Kiến thức cơ bản cần thiết](#kiến-thức-cơ-bản-cần-thiết)
3. [Kiến thức theo từng module](#kiến-thức-theo-từng-module)
4. [Lộ trình học tập](#lộ-trình-học-tập)
5. [Tài liệu tham khảo](#tài-liệu-tham-khảo)
6. [Thực hành và dự án](#thực-hành-và-dự-án)

---

## 🎯 Tổng quan dự án

### Dự án này bao gồm:

1. **Edge AI Application** - Ứng dụng xử lý AI trên thiết bị edge (camera)
2. **Backend API** - API server xử lý dữ liệu và analytics
3. **Frontend Dashboard** - Giao diện web quản lý và hiển thị dữ liệu
4. **AI Models** - Mô hình deep learning cho face detection, age/gender/emotion

### Tech Stack chính:

- **Python** (Backend, Edge AI)
- **React + TypeScript** (Frontend)
- **FastAPI** (Backend Framework)
- **PyTorch/ONNX** (AI Models)
- **OpenCV** (Computer Vision)
- **PostgreSQL/SQLite** (Database)
- **MQTT** (Messaging)
- **Docker** (Containerization)

---

## 📖 Kiến thức cơ bản cần thiết

### 1. Python Programming (Bắt buộc)

#### Cấp độ: Cơ bản → Trung cấp

**Kiến thức cần nắm:**

- ✅ **Cú pháp Python cơ bản**
  - Variables, Data types (int, float, string, list, dict, tuple)
  - Control flow (if/else, for, while)
  - Functions và Lambda functions
  - List/Dict comprehensions
  - Exception handling (try/except/finally)

- ✅ **OOP trong Python**
  - Classes và Objects
  - Inheritance (Kế thừa)
  - Encapsulation (Đóng gói)
  - Polymorphism (Đa hình)
  - Magic methods (`__init__`, `__str__`, `__repr__`)

- ✅ **Python Standard Library**
  - `os`, `sys`, `pathlib` - File system operations
  - `json` - JSON parsing
  - `threading`, `multiprocessing` - Concurrency
  - `collections` - Advanced data structures
  - `typing` - Type hints
  - `logging` - Logging system

- ✅ **Package Management**
  - `pip` - Package installer
  - `requirements.txt` - Dependency management
  - Virtual environments (`venv`, `conda`)

**Tài liệu học:**
- 📚 **Python Official Tutorial**: https://docs.python.org/3/tutorial/
- 📚 **Real Python**: https://realpython.com/
- 📚 **Python Crash Course** (Book) - Eric Matthes
- 🎥 **Python for Everybody** (Coursera) - Free course

**Thời gian ước tính:** 2-3 tháng (nếu học từ đầu)

---

### 2. Web Development Fundamentals

#### 2.1. HTML/CSS/JavaScript (Frontend)

**Kiến thức cần nắm:**

- ✅ **HTML5**
  - Semantic HTML
  - Forms và Input types
  - Accessibility basics

- ✅ **CSS3**
  - Flexbox và Grid Layout
  - CSS Variables
  - Responsive Design (Media Queries)
  - CSS Animations
  - Dark Mode implementation

- ✅ **JavaScript (ES6+)**
  - Variables (`let`, `const`, `var`)
  - Functions (Arrow functions, Callbacks)
  - Arrays methods (map, filter, reduce)
  - Promises và Async/Await
  - DOM Manipulation
  - Event Handling
  - Fetch API

**Tài liệu học:**
- 📚 **MDN Web Docs**: https://developer.mozilla.org/
- 📚 **JavaScript.info**: https://javascript.info/
- 🎥 **JavaScript Crash Course** (YouTube)

**Thời gian ước tính:** 1-2 tháng

---

#### 2.2. React Framework (Frontend)

**Kiến thức cần nắm:**

- ✅ **React Core Concepts**
  - Components (Functional & Class)
  - Props và State
  - Hooks (`useState`, `useEffect`, `useContext`, `useReducer`)
  - Event Handling
  - Conditional Rendering
  - Lists và Keys

- ✅ **React Advanced**
  - Custom Hooks
  - Context API
  - Performance Optimization (useMemo, useCallback)
  - Error Boundaries
  - React Router (Routing)

- ✅ **State Management**
  - Redux Toolkit
  - Redux Store, Actions, Reducers
  - Async Actions (Thunks)
  - Redux DevTools

- ✅ **Build Tools**
  - Vite (Build tool)
  - npm/yarn (Package manager)
  - Environment Variables

**Tài liệu học:**
- 📚 **React Official Docs**: https://react.dev/
- 📚 **Redux Toolkit Docs**: https://redux-toolkit.js.org/
- 🎥 **React - The Complete Guide** (Udemy)
- 🎥 **Redux Toolkit Tutorial** (YouTube)

**Thời gian ước tính:** 2-3 tháng

---

### 3. Backend Development

#### 3.1. FastAPI Framework

**Kiến thức cần nắm:**

- ✅ **FastAPI Basics**
  - Routing và HTTP Methods (GET, POST, PUT, DELETE)
  - Path Parameters và Query Parameters
  - Request Body (Pydantic Models)
  - Response Models
  - Status Codes

- ✅ **FastAPI Advanced**
  - Dependency Injection
  - Authentication (JWT Tokens)
  - CORS (Cross-Origin Resource Sharing)
  - WebSockets
  - Background Tasks
  - Middleware

- ✅ **Pydantic**
  - Data Validation
  - Model Definition
  - Field Validation
  - Custom Validators

- ✅ **Database Integration**
  - SQLAlchemy ORM
  - Database Models
  - Relationships (One-to-Many, Many-to-Many)
  - Database Migrations
  - Connection Pooling

**Tài liệu học:**
- 📚 **FastAPI Official Docs**: https://fastapi.tiangolo.com/
- 📚 **SQLAlchemy Docs**: https://docs.sqlalchemy.org/
- 🎥 **FastAPI Tutorial** (YouTube)

**Thời gian ước tính:** 1-2 tháng

---

#### 3.2. RESTful API Design

**Kiến thức cần nắm:**

- ✅ **HTTP Protocol**
  - HTTP Methods (GET, POST, PUT, DELETE, PATCH)
  - Status Codes (200, 201, 400, 401, 404, 500)
  - Headers và Body
  - Content-Type (JSON, Form Data)

- ✅ **API Design Principles**
  - RESTful conventions
  - Resource naming
  - Versioning (`/api/v1/`)
  - Pagination
  - Filtering và Sorting

- ✅ **Authentication & Authorization**
  - JWT (JSON Web Tokens)
  - OAuth2
  - Password Hashing (bcrypt)
  - Session Management

**Tài liệu học:**
- 📚 **REST API Tutorial**: https://restfulapi.net/
- 📚 **JWT.io**: https://jwt.io/

**Thời gian ước tính:** 2-3 tuần

---

### 4. Database

#### 4.1. SQL Fundamentals

**Kiến thức cần nắm:**

- ✅ **SQL Basics**
  - SELECT, INSERT, UPDATE, DELETE
  - WHERE, ORDER BY, GROUP BY
  - JOINs (INNER, LEFT, RIGHT, FULL)
  - Aggregation Functions (COUNT, SUM, AVG, MAX, MIN)
  - Subqueries

- ✅ **Database Design**
  - Normalization (1NF, 2NF, 3NF)
  - Primary Keys, Foreign Keys
  - Indexes
  - Relationships

**Tài liệu học:**
- 📚 **SQL Tutorial**: https://www.w3schools.com/sql/
- 📚 **PostgreSQL Tutorial**: https://www.postgresql.org/docs/
- 🎥 **SQL for Beginners** (YouTube)

**Thời gian ước tính:** 1 tháng

---

#### 4.2. PostgreSQL

**Kiến thức cần nắm:**

- ✅ **PostgreSQL Basics**
  - Installation và Setup
  - Creating Databases và Tables
  - Data Types
  - Constraints

- ✅ **Advanced Features**
  - Transactions
  - Stored Procedures
  - Triggers
  - Views
  - Full-text Search

**Tài liệu học:**
- 📚 **PostgreSQL Official Docs**: https://www.postgresql.org/docs/

**Thời gian ước tính:** 2-3 tuần

---

### 5. Computer Vision & Deep Learning

#### 5.1. OpenCV (Computer Vision)

**Kiến thức cần nắm:**

- ✅ **OpenCV Basics**
  - Image Reading/Writing
  - Image Manipulation (Resize, Crop, Rotate)
  - Color Spaces (RGB, HSV, Grayscale)
  - Image Filtering (Blur, Sharpen)
  - Edge Detection (Canny)

- ✅ **Video Processing**
  - Video Capture (`cv2.VideoCapture`)
  - Frame Processing
  - Video Writing
  - Camera Access

- ✅ **Object Detection**
  - Haar Cascades
  - DNN (Deep Neural Networks)
  - YOLO Integration
  - Face Detection

**Tài liệu học:**
- 📚 **OpenCV Official Docs**: https://docs.opencv.org/
- 📚 **OpenCV Python Tutorials**: https://opencv-python-tutroals.readthedocs.io/
- 🎥 **OpenCV Course** (YouTube - freeCodeCamp)

**Thời gian ước tính:** 1-2 tháng

---

#### 5.2. Deep Learning Fundamentals

**Kiến thức cần nắm:**

- ✅ **Neural Networks Basics**
  - Perceptron
  - Multi-layer Perceptron (MLP)
  - Activation Functions (ReLU, Sigmoid, Tanh)
  - Loss Functions (Cross-Entropy, MSE)
  - Backpropagation
  - Gradient Descent

- ✅ **Convolutional Neural Networks (CNN)**
  - Convolution Layers
  - Pooling Layers (Max, Average)
  - Fully Connected Layers
  - CNN Architectures (LeNet, AlexNet, VGG, ResNet)

- ✅ **Transfer Learning**
  - Pre-trained Models
  - Fine-tuning
  - Feature Extraction

**Tài liệu học:**
- 📚 **Deep Learning Book** (Ian Goodfellow): https://www.deeplearningbook.org/
- 📚 **Neural Networks and Deep Learning** (Michael Nielsen): http://neuralnetworksanddeeplearning.com/
- 🎥 **Deep Learning Specialization** (Coursera - Andrew Ng)
- 🎥 **Fast.ai Course**: https://www.fast.ai/

**Thời gian ước tính:** 3-4 tháng

---

#### 5.3. PyTorch Framework

**Kiến thức cần nắm:**

- ✅ **PyTorch Basics**
  - Tensors (Creation, Operations)
  - Autograd (Automatic Differentiation)
  - Neural Network Module (`nn.Module`)
  - Loss Functions và Optimizers
  - Training Loop

- ✅ **PyTorch Advanced**
  - Data Loading (`DataLoader`, `Dataset`)
  - Transfer Learning với `torchvision`
  - Model Saving/Loading
  - Mixed Precision Training (AMP)
  - GPU Acceleration (CUDA)

- ✅ **Model Deployment**
  - ONNX Export
  - ONNX Runtime
  - Model Optimization

**Tài liệu học:**
- 📚 **PyTorch Official Tutorials**: https://pytorch.org/tutorials/
- 📚 **PyTorch Documentation**: https://pytorch.org/docs/
- 🎥 **PyTorch for Deep Learning** (YouTube - freeCodeCamp)

**Thời gian ước tính:** 2-3 tháng

---

### 6. Object Tracking & Detection

**Kiến thức cần nắm:**

- ✅ **Object Detection**
  - YOLO (You Only Look Once)
  - RetinaFace
  - Bounding Boxes
  - Non-Maximum Suppression (NMS)
  - IoU (Intersection over Union)

- ✅ **Object Tracking**
  - Kalman Filter
  - DeepSORT Algorithm
  - ByteTrack Algorithm
  - Multi-Object Tracking (MOT)
  - Track Association

**Tài liệu học:**
- 📚 **YOLO Paper**: https://arxiv.org/abs/1506.02640
- 📚 **DeepSORT Paper**: https://arxiv.org/abs/1703.07402
- 📚 **ByteTrack Paper**: https://arxiv.org/abs/2110.06864
- 🎥 **Object Tracking Tutorials** (YouTube)

**Thời gian ước tính:** 1-2 tháng

---

### 7. MQTT & Messaging

**Kiến thức cần nắm:**

- ✅ **MQTT Protocol**
  - MQTT Basics (Broker, Client, Topics)
  - Publish/Subscribe Pattern
  - QoS Levels (0, 1, 2)
  - Retained Messages
  - Last Will and Testament

- ✅ **paho-mqtt Library**
  - Client Connection
  - Publishing Messages
  - Subscribing to Topics
  - Callbacks

**Tài liệu học:**
- 📚 **MQTT.org**: https://mqtt.org/
- 📚 **Eclipse Paho MQTT**: https://www.eclipse.org/paho/
- 🎥 **MQTT Tutorial** (YouTube)

**Thời gian ước tính:** 1 tuần

---

### 8. Docker & Containerization

**Kiến thức cần nắm:**

- ✅ **Docker Basics**
  - Docker Images và Containers
  - Dockerfile
  - Docker Compose
  - Volume Mounting
  - Port Mapping

- ✅ **Docker Advanced**
  - Multi-stage Builds
  - Docker Networking
  - Docker Registry
  - Best Practices

**Tài liệu học:**
- 📚 **Docker Official Docs**: https://docs.docker.com/
- 🎥 **Docker Tutorial** (YouTube - freeCodeCamp)

**Thời gian ước tính:** 1-2 tuần

---

## 🔧 Kiến thức theo từng module

### Module 1: Edge AI Application

**Các file chính:**
- `main.py` / `main_gui.py` - Entry point
- `src/detectors/` - Face detection
- `src/trackers/` - Object tracking
- `src/classifiers/` - Age/Gender/Emotion classification
- `src/ads_engine/` - Advertisement recommendation

**Kiến thức cần:**
1. ✅ OpenCV - Video capture và processing
2. ✅ PyTorch/ONNX - Model inference
3. ✅ Multi-threading - Performance optimization
4. ✅ Object Detection (YOLO, RetinaFace)
5. ✅ Object Tracking (DeepSORT, ByteTrack)
6. ✅ PyQt6 - GUI development (cho GUI version)

**Học theo thứ tự:**
1. OpenCV basics → Video processing
2. PyTorch → Model loading và inference
3. Object Detection → YOLO/RetinaFace
4. Object Tracking → DeepSORT/ByteTrack
5. Multi-threading → Performance optimization
6. PyQt6 → GUI development

---

### Module 2: Backend API

**Các file chính:**
- `app/main.py` - FastAPI application
- `app/api/` - API endpoints
- `app/models/` - Database models
- `app/services/` - Business logic
- `app/core/` - Core utilities

**Kiến thức cần:**
1. ✅ FastAPI - Web framework
2. ✅ SQLAlchemy - ORM
3. ✅ PostgreSQL/SQLite - Database
4. ✅ JWT Authentication - Security
5. ✅ Pydantic - Data validation
6. ✅ WebSockets - Real-time communication

**Học theo thứ tự:**
1. FastAPI basics → Routing, Request/Response
2. SQLAlchemy → Database models và queries
3. Authentication → JWT tokens
4. WebSockets → Real-time updates
5. API Design → RESTful principles

---

### Module 3: Frontend Dashboard

**Các file chính:**
- `src/App.jsx` - Main app component
- `src/pages/` - Page components
- `src/components/` - Reusable components
- `src/store/` - Redux state management
- `src/services/` - API services

**Kiến thức cần:**
1. ✅ React - UI framework
2. ✅ Redux Toolkit - State management
3. ✅ Ant Design - UI components
4. ✅ React Router - Navigation
5. ✅ Axios/Fetch - API calls
6. ✅ CSS/Tailwind - Styling

**Học theo thứ tự:**
1. React basics → Components, Hooks
2. Redux Toolkit → State management
3. React Router → Navigation
4. Ant Design → UI components
5. API Integration → Fetching data
6. Styling → CSS/Tailwind

---

### Module 4: AI Models Training

**Các file chính:**
- `training_experiments/notebooks/` - Training notebooks
- `training_experiments/models/` - Trained models

**Kiến thức cần:**
1. ✅ PyTorch - Deep learning framework
2. ✅ Data Loading - Dataset preparation
3. ✅ Training Loop - Model training
4. ✅ Transfer Learning - Pre-trained models
5. ✅ Model Evaluation - Metrics và validation
6. ✅ Model Export - ONNX conversion

**Học theo thứ tự:**
1. PyTorch basics → Tensors, Autograd
2. CNN Architectures → ResNet, EfficientNet
3. Transfer Learning → Fine-tuning
4. Training Pipeline → Data, Training, Validation
5. Model Optimization → Hyperparameter tuning
6. Model Deployment → ONNX export

---

## 🗺️ Lộ trình học tập

### Lộ trình 6 tháng (Full-time)

#### Tháng 1-2: Foundation
- ✅ Python Programming (Cơ bản → Trung cấp)
- ✅ HTML/CSS/JavaScript
- ✅ SQL Fundamentals
- ✅ Git/GitHub

**Dự án thực hành:**
- Tạo một web app đơn giản với Python Flask
- Tạo một dashboard với HTML/CSS/JS

---

#### Tháng 3: Backend Development
- ✅ FastAPI Framework
- ✅ SQLAlchemy
- ✅ RESTful API Design
- ✅ Authentication (JWT)

**Dự án thực hành:**
- Tạo một REST API với FastAPI
- Tích hợp database PostgreSQL
- Implement authentication

---

#### Tháng 4: Frontend Development
- ✅ React Framework
- ✅ Redux Toolkit
- ✅ React Router
- ✅ Ant Design

**Dự án thực hành:**
- Tạo một React dashboard
- Tích hợp với Backend API
- Implement state management với Redux

---

#### Tháng 5: Computer Vision & AI
- ✅ OpenCV
- ✅ Deep Learning Fundamentals
- ✅ PyTorch
- ✅ CNN Architectures

**Dự án thực hành:**
- Face detection với OpenCV
- Train một CNN model với PyTorch
- Object detection với YOLO

---

#### Tháng 6: Advanced Topics
- ✅ Object Tracking (DeepSORT, ByteTrack)
- ✅ Model Deployment (ONNX)
- ✅ MQTT Messaging
- ✅ Docker

**Dự án thực hành:**
- Implement object tracking
- Deploy model với ONNX Runtime
- Tích hợp MQTT messaging

---

### Lộ trình 12 tháng (Part-time)

**6 tháng đầu:** Foundation + Backend + Frontend
**6 tháng sau:** Computer Vision + AI + Advanced Topics

---

## 📚 Tài liệu tham khảo

### Sách

1. **Python**
   - "Python Crash Course" - Eric Matthes
   - "Fluent Python" - Luciano Ramalho

2. **Web Development**
   - "Full Stack React" - Anthony Accomazzo
   - "You Don't Know JS" - Kyle Simpson

3. **Deep Learning**
   - "Deep Learning" - Ian Goodfellow
   - "Hands-On Machine Learning" - Aurélien Géron

4. **Computer Vision**
   - "Learning OpenCV" - Gary Bradski
   - "Computer Vision: Algorithms and Applications" - Richard Szeliski

---

### Khóa học Online

#### Free Courses

1. **Python**
   - Python for Everybody (Coursera) - Free
   - Python Crash Course (freeCodeCamp YouTube)

2. **Web Development**
   - The Odin Project - Free
   - freeCodeCamp - Free

3. **Deep Learning**
   - Fast.ai - Free
   - Deep Learning Specialization (Coursera) - Free audit

4. **Computer Vision**
   - OpenCV Course (freeCodeCamp YouTube)
   - PyTorch Tutorials (Official)

---

#### Paid Courses (Khuyến nghị)

1. **Udemy**
   - "Complete Python Bootcamp" - Jose Portilla
   - "React - The Complete Guide" - Maximilian Schwarzmüller
   - "FastAPI - The Complete Course" - Various

2. **Coursera**
   - "Deep Learning Specialization" - Andrew Ng
   - "Machine Learning" - Andrew Ng

3. **Pluralsight**
   - Various Python, React, FastAPI courses

---

### YouTube Channels

1. **freeCodeCamp** - Full courses
2. **Corey Schafer** - Python tutorials
3. **Traversy Media** - Web development
4. **Sentdex** - Python, Machine Learning
5. **3Blue1Brown** - Deep Learning explained

---

### Documentation

1. **Official Docs**
   - Python: https://docs.python.org/3/
   - React: https://react.dev/
   - FastAPI: https://fastapi.tiangolo.com/
   - PyTorch: https://pytorch.org/docs/
   - OpenCV: https://docs.opencv.org/

2. **Community Resources**
   - Stack Overflow
   - GitHub Discussions
   - Reddit (r/learnpython, r/reactjs, r/MachineLearning)

---

## 💻 Thực hành và dự án

### Dự án theo cấp độ

#### Beginner Projects

1. **Python CLI App**
   - Tạo một command-line tool
   - File I/O, JSON parsing

2. **Simple Web API**
   - FastAPI với CRUD operations
   - SQLite database

3. **React Todo App**
   - Basic React components
   - State management với useState

---

#### Intermediate Projects

1. **Face Detection App**
   - OpenCV face detection
   - Webcam integration
   - GUI với PyQt6

2. **Full Stack Dashboard**
   - React frontend
   - FastAPI backend
   - PostgreSQL database
   - Authentication

3. **Image Classification**
   - Train CNN với PyTorch
   - Transfer learning
   - Model deployment

---

#### Advanced Projects

1. **Object Tracking System**
   - YOLO detection
   - DeepSORT tracking
   - Real-time processing

2. **Smart Retail System** (Dự án này!)
   - Edge AI processing
   - Backend API
   - Frontend Dashboard
   - Real-time analytics

---

### Tips học tập hiệu quả

1. **Học bằng cách làm (Learning by Doing)**
   - Đọc code → Hiểu → Viết lại
   - Thực hành ngay sau khi học lý thuyết

2. **Break down problems**
   - Chia nhỏ vấn đề
   - Giải quyết từng phần

3. **Read documentation**
   - Đọc official docs thay vì chỉ tutorials
   - Hiểu sâu hơn về framework/library

4. **Join communities**
   - Stack Overflow
   - GitHub Discussions
   - Discord/Slack communities

5. **Build projects**
   - Apply kiến thức vào dự án thực tế
   - Portfolio để showcase

---

## 🎯 Checklist học tập

### Foundation (Tháng 1-2)
- [ ] Python cơ bản (Variables, Functions, OOP)
- [ ] HTML/CSS/JavaScript
- [ ] SQL queries
- [ ] Git/GitHub

### Backend (Tháng 3)
- [ ] FastAPI routing
- [ ] Database với SQLAlchemy
- [ ] Authentication với JWT
- [ ] API design principles

### Frontend (Tháng 4)
- [ ] React components và hooks
- [ ] Redux state management
- [ ] React Router
- [ ] API integration

### Computer Vision (Tháng 5)
- [ ] OpenCV basics
- [ ] Video processing
- [ ] Face detection
- [ ] Object detection

### Deep Learning (Tháng 5-6)
- [ ] Neural networks fundamentals
- [ ] CNN architectures
- [ ] PyTorch framework
- [ ] Model training

### Advanced (Tháng 6)
- [ ] Object tracking
- [ ] Model deployment (ONNX)
- [ ] MQTT messaging
- [ ] Docker containerization

---

## 📞 Hỗ trợ và Resources

### Khi gặp vấn đề:

1. **Đọc Error Messages**
   - Error messages thường chỉ ra vấn đề
   - Stack trace cho biết vị trí lỗi

2. **Google Search**
   - Copy error message
   - Tìm trên Stack Overflow

3. **Documentation**
   - Đọc official docs
   - Tìm examples

4. **Ask for Help**
   - Stack Overflow
   - GitHub Issues
   - Community forums

---

## 🔄 Cập nhật

File này sẽ được cập nhật khi:
- Có thêm technologies mới vào project
- Có resources học tập tốt hơn
- Có feedback từ người học

**Last Updated:** 2024-12-31

---

## 📝 Ghi chú

- **Thời gian ước tính** là cho người học từ đầu, có thể nhanh hơn nếu đã có background
- **Lộ trình** có thể điều chỉnh theo nhu cầu và thời gian
- **Quan trọng nhất**: Thực hành nhiều, đọc code, viết code

**Chúc bạn học tập thành công! 🚀**
