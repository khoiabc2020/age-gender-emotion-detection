import React, { useState, useEffect } from 'react'
import { Form, Input, Button, Card, message, Typography, Divider } from 'antd'
import { UserOutlined, LockOutlined, MailOutlined } from '@ant-design/icons'
import { useNavigate, Link } from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../store/hooks'
import { login as loginThunk } from '../store/slices/authSlice'
import api from '../services/api'

const { Title, Text } = Typography

const LoginPage = () => {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const dispatch = useAppDispatch()
  const isAuthenticated = useAppSelector((state) => state.auth.isAuthenticated)
  const authError = useAppSelector((state) => state.auth.error)

  // Navigate when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/')
    }
  }, [isAuthenticated, navigate])

  // Show error message
  useEffect(() => {
    if (authError) {
      message.error(authError)
    }
  }, [authError])

  const onFinish = async (values) => {
    setLoading(true)
    try {
      // Use async thunk - it handles API call and state update
      const result = await dispatch(loginThunk({
        username: values.username,
        password: values.password
      }))
      
      if (loginThunk.fulfilled.match(result)) {
        message.success('Đăng nhập thành công!')
        // Navigation will happen automatically via useEffect
      } else {
        const errorMsg = result.payload || 'Đăng nhập thất bại!'
        message.error(errorMsg)
      }
    } catch (error) {
      message.error('Đã xảy ra lỗi khi đăng nhập!')
      console.error('Login error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Background decoration */}
      <div style={{
        position: 'absolute',
        top: '-50%',
        right: '-50%',
        width: '200%',
        height: '200%',
        background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
        animation: 'pulse 20s ease-in-out infinite'
      }} />
      
      <Card
        style={{
          width: '100%',
          maxWidth: '440px',
          borderRadius: '16px',
          boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
          border: 'none',
          background: '#ffffff',
          position: 'relative',
          zIndex: 1
        }}
        bodyStyle={{ padding: '48px' }}
      >
        {/* Logo/Title */}
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <div style={{
            width: '64px',
            height: '64px',
            margin: '0 auto 20px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 8px 24px rgba(102, 126, 234, 0.4)'
          }}>
            <span style={{ fontSize: '32px', color: '#ffffff', fontWeight: 'bold' }}>S</span>
          </div>
          <Title level={2} style={{ marginBottom: '8px', color: '#262626', fontWeight: 600 }}>
            Smart Retail
          </Title>
          <Text style={{ fontSize: '15px', color: '#8c8c8c' }}>
            Đăng nhập vào tài khoản của bạn
          </Text>
        </div>

        {/* Login Form */}
        <Form
          name="login"
          onFinish={onFinish}
          autoComplete="off"
          size="large"
          layout="vertical"
        >
          <Form.Item
            name="username"
            rules={[
              { required: true, message: 'Vui lòng nhập tên đăng nhập!' },
            ]}
          >
            <Input
              prefix={<UserOutlined style={{ color: '#bfbfbf' }} />}
              placeholder="Tên đăng nhập"
              size="large"
              style={{ 
                borderRadius: '8px',
                height: '48px',
                fontSize: '15px'
              }}
            />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[
              { required: true, message: 'Vui lòng nhập mật khẩu!' },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
              placeholder="Mật khẩu"
              size="large"
              style={{ 
                borderRadius: '8px',
                height: '48px',
                fontSize: '15px'
              }}
            />
          </Form.Item>

          <Form.Item style={{ marginTop: '24px', marginBottom: 0 }}>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              block
              size="large"
              style={{
                height: '48px',
                borderRadius: '8px',
                fontSize: '16px',
                fontWeight: 600,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none',
                boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)'
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.5)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.4)'
              }}
            >
              Đăng nhập
            </Button>
          </Form.Item>
        </Form>

        <Divider style={{ margin: '32px 0' }}>
          <Text style={{ fontSize: '13px', color: '#bfbfbf' }}>hoặc</Text>
        </Divider>

        {/* Register Link */}
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <Text style={{ fontSize: '14px', color: '#8c8c8c' }}>
            Chưa có tài khoản?{' '}
            <Link 
              to="/register" 
              style={{ 
                color: '#667eea', 
                fontWeight: 600,
                textDecoration: 'none'
              }}
            >
              Đăng ký ngay
            </Link>
          </Text>
        </div>

        {/* Default Credentials Hint */}
        <div style={{
          padding: '16px',
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          borderRadius: '8px',
          textAlign: 'center',
          border: '1px solid #e8e8e8'
        }}>
          <Text style={{ fontSize: '13px', color: '#595959', fontWeight: 500 }}>
            Demo: admin / admin123
          </Text>
        </div>
      </Card>
    </div>
  )
}

export default LoginPage
