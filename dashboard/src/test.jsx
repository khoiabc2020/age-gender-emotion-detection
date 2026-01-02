import React from 'react'
import ReactDOM from 'react-dom/client'

// Simple test page to check if React is working
function TestApp() {
  return (
    <div style={{ 
      padding: '50px', 
      fontFamily: 'Arial', 
      textAlign: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      minHeight: '100vh',
      color: 'white'
    }}>
      <h1 style={{ fontSize: '48px', marginBottom: '20px' }}>
        ✅ Frontend đang chạy!
      </h1>
      <p style={{ fontSize: '24px', marginBottom: '40px' }}>
        React và Vite hoạt động bình thường
      </p>
      <div style={{ 
        background: 'rgba(255,255,255,0.2)', 
        padding: '30px', 
        borderRadius: '10px',
        maxWidth: '600px',
        margin: '0 auto'
      }}>
        <h2>Thông tin hệ thống:</h2>
        <p>React Version: {React.version}</p>
        <p>Time: {new Date().toLocaleString('vi-VN')}</p>
        <p>URL: {window.location.href}</p>
      </div>
      <div style={{ marginTop: '40px' }}>
        <a 
          href="/login" 
          style={{
            color: 'white',
            textDecoration: 'underline',
            fontSize: '18px'
          }}
        >
          → Đi tới trang Login
        </a>
      </div>
    </div>
  )
}

const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(<TestApp />)
