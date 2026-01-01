# Dashboard

React + Vite based web dashboard for Smart Retail AI analytics and monitoring.

---

## 🚀 **Features**

- **Real-time Analytics**: Live customer data visualization
- **AI Agent Chat**: Integrated Google AI & ChatGPT
- **User Management**: Authentication and authorization
- **Data Visualization**: Charts, graphs, and statistics
- **Advertisement Management**: Configure ad rules
- **Responsive Design**: Mobile-friendly interface
- **Dark Mode**: Eye-friendly theme
- **WebSocket Support**: Real-time updates

---

## 📁 **Structure**

```
dashboard/
├── src/
│   ├── App.jsx              # Main application
│   ├── main.jsx             # Entry point
│   ├── index.css            # Global styles
│   ├── components/          # React components
│   │   ├── Dashboard.jsx    # Main dashboard
│   │   ├── Analytics.jsx    # Analytics view
│   │   ├── AIAgent.jsx      # AI chat interface
│   │   ├── Login.jsx        # Login page
│   │   └── ... (17 components)
│   └── utils/               # Utility functions
├── public/                  # Static assets
├── package.json             # Dependencies
├── vite.config.js          # Build config
├── tailwind.config.js      # Tailwind CSS
└── Dockerfile              # Container config
```

---

## 🛠️ **Setup**

### **Prerequisites:**
- Node.js 18+ & npm/yarn
- Backend API running (http://localhost:8000)

### **Installation:**

```bash
cd dashboard

# Install dependencies
npm install
# or
yarn install

# Setup environment
cp ../.env.example .env
# Edit .env with your values
```

---

## 🚀 **Run**

### **Development:**

```bash
# Start dev server with hot reload
npm run dev
# or
yarn dev
```

**Access:** http://localhost:3000

### **Production Build:**

```bash
# Build for production
npm run build
# or
yarn build

# Preview production build
npm run preview
```

### **Docker:**

```bash
docker build -t smartretail-dashboard .
docker run -p 3000:80 smartretail-dashboard
```

---

## 🔑 **Default Login**

**Development:**
- Username: `admin`
- Password: `admin123`

⚠️ **Change in production!**

---

## 📊 **Components**

### **Main Components:**

**Dashboard.jsx**
- Overview statistics
- Quick access cards
- System status

**Analytics.jsx**
- Customer demographics
- Emotion distribution
- Time-based analytics
- Interactive charts

**AIAgent.jsx**
- Chat interface
- Conversation history
- AI response streaming
- Report generation

**Login.jsx**
- User authentication
- Form validation
- JWT token management

### **Utility Components:**

- **ProtectedRoute**: Route protection
- **Navbar**: Navigation bar
- **Sidebar**: Side navigation
- **Chart components**: Various charts
- **Modal**: Popup dialogs

---

## 🎨 **Styling**

### **Tailwind CSS:**

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: '#3b82f6',
        secondary: '#8b5cf6',
        // ... custom colors
      }
    }
  }
}
```

### **Custom Styles:**

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Custom utilities */
.glass-effect {
  backdrop-filter: blur(10px);
  background: rgba(255, 255, 255, 0.1);
}
```

---

## 🔌 **API Integration**

### **Configuration:**

```javascript
// vite.config.js
export default {
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
}
```

### **API Calls:**

```javascript
// Example: Fetch analytics data
const response = await fetch('/api/v1/analytics/stats', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
const data = await response.json();
```

---

## 🧪 **Testing**

```bash
# Run tests
npm test
# or
yarn test

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

---

## 📦 **Build**

### **Development Build:**

```bash
npm run dev
```

### **Production Build:**

```bash
# Build optimized bundle
npm run build

# Output: dist/
# - Minified JavaScript
# - Optimized CSS
# - Compressed assets
```

### **Build Size:**

```
dist/
├── assets/
│   ├── index-[hash].js     (~150 KB gzipped)
│   └── index-[hash].css    (~20 KB gzipped)
└── index.html              (~2 KB)
```

---

## 🔧 **Configuration**

### **Environment Variables:**

```env
# .env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_ENV=development
```

### **Vite Config:**

```javascript
// vite.config.js
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser'
  }
})
```

---

## 🐛 **Troubleshooting**

### **CORS Issues:**

```javascript
// Check VITE_API_URL in .env
// Ensure backend allows your frontend URL in CORS settings
```

### **WebSocket Connection:**

```javascript
// Check VITE_WS_URL
// Ensure backend WebSocket endpoint is accessible
```

### **Build Errors:**

```bash
# Clear cache
rm -rf node_modules
rm package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
npm run dev
```

---

## 📊 **Features Detail**

### **Real-time Updates:**

```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateDashboard(data);
};
```

### **Charts:**

Using **Chart.js** and **React-Chartjs-2**:

```javascript
import { Line, Bar, Pie } from 'react-chartjs-2';

<Line data={chartData} options={chartOptions} />
```

### **AI Agent:**

```javascript
// Chat with AI
const response = await fetch('/api/v1/ai/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    message: userMessage,
    provider: 'google_ai'
  })
});
```

---

## 🎨 **UI Components**

### **Technology Stack:**

- **React 18**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Chart.js**: Data visualization
- **React Router**: Navigation
- **Axios**: HTTP client
- **WebSocket**: Real-time communication

---

## 📱 **Responsive Design**

### **Breakpoints:**

```css
/* Tailwind CSS breakpoints */
sm: 640px   /* Mobile landscape */
md: 768px   /* Tablet */
lg: 1024px  /* Desktop */
xl: 1280px  /* Large desktop */
2xl: 1536px /* Extra large */
```

### **Mobile-First:**

```jsx
<div className="
  w-full           // Mobile
  md:w-1/2         // Tablet: 50% width
  lg:w-1/3         // Desktop: 33% width
">
  Content
</div>
```

---

## 🚀 **Deployment**

### **Docker:**

```bash
docker-compose up -d dashboard
```

### **Nginx:**

```nginx
server {
    listen 80;
    server_name dashboard.example.com;
    
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://backend:8000;
    }
}
```

### **Kubernetes:**

```bash
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/dashboard-service.yaml
```

---

## 📚 **Documentation**

- **React Docs**: https://react.dev
- **Vite Docs**: https://vitejs.dev
- **Tailwind CSS**: https://tailwindcss.com
- **Chart.js**: https://www.chartjs.org

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
**Live Demo:** Coming soon
