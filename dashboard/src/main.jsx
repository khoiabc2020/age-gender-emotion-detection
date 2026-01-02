import React from 'react'
import ReactDOM from 'react-dom/client'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import viVN from 'antd/locale/vi_VN'
import dayjs from 'dayjs'
import 'dayjs/locale/vi'
import App from './App'
import { store } from './store/store'
import { ThemeProvider } from './components/layout/ThemeProvider'
import './index.css'

dayjs.locale('vi')

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('React Error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '50px', 
          textAlign: 'center',
          fontFamily: 'Arial',
          background: '#f0f0f0',
          minHeight: '100vh'
        }}>
          <h1 style={{ color: '#e74c3c', fontSize: '32px' }}>⚠️ Lỗi ứng dụng</h1>
          <p style={{ color: '#666', fontSize: '18px', marginTop: '20px' }}>
            {this.state.error?.message || 'Đã xảy ra lỗi không mong muốn'}
          </p>
          <button 
            onClick={() => window.location.reload()}
            style={{
              marginTop: '30px',
              padding: '12px 24px',
              fontSize: '16px',
              background: '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            Tải lại trang
          </button>
          <details style={{ marginTop: '30px', textAlign: 'left', maxWidth: '800px', margin: '30px auto' }}>
            <summary style={{ cursor: 'pointer', color: '#667eea' }}>Chi tiết lỗi</summary>
            <pre style={{ 
              background: '#fff', 
              padding: '20px', 
              borderRadius: '6px',
              overflow: 'auto',
              marginTop: '10px'
            }}>
              {this.state.error?.stack || JSON.stringify(this.state.error, null, 2)}
            </pre>
          </details>
        </div>
      )
    }

    return this.props.children
  }
}

// Main render with error handling
try {
  const rootElement = document.getElementById('root')
  if (!rootElement) {
    throw new Error('Root element not found!')
  }

  const root = ReactDOM.createRoot(rootElement)
  
  root.render(
    <React.StrictMode>
      <ErrorBoundary>
        <Provider store={store}>
          <BrowserRouter>
            <ConfigProvider locale={viVN}>
              <ThemeProvider>
                <App />
              </ThemeProvider>
            </ConfigProvider>
          </BrowserRouter>
        </Provider>
      </ErrorBoundary>
    </React.StrictMode>
  )
  
  console.log('✅ React app rendered successfully!')
} catch (error) {
  console.error('❌ Failed to render React app:', error)
  document.body.innerHTML = `
    <div style="padding: 50px; text-align: center; font-family: Arial;">
      <h1 style="color: #e74c3c;">❌ Lỗi khởi động ứng dụng</h1>
      <p style="color: #666; margin-top: 20px;">${error.message}</p>
      <pre style="background: #f0f0f0; padding: 20px; border-radius: 6px; margin-top: 20px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;">
        ${error.stack}
      </pre>
    </div>
  `
}
