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
      background: '#f5f5f5',
      padding: '20px'
    }}>
      <Card
        style={{
          width: '100%',
          maxWidth: '420px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          border: '1px solid #e8e8e8'
        }}
        bodyStyle={{ padding: '40px' }}
      >
        {/* Logo/Title */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <Title level={2} style={{ marginBottom: '8px', color: '#262626' }}>
            Smart Retail
          </Title>
          <Text type="secondary" style={{ fontSize: '14px' }}>
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
              style={{ borderRadius: '4px' }}
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
              style={{ borderRadius: '4px' }}
            />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              block
              style={{
                height: '44px',
                borderRadius: '4px',
                fontSize: '16px',
                fontWeight: 500,
                marginTop: '8px'
              }}
            >
              Đăng nhập
            </Button>
          </Form.Item>
        </Form>

        <Divider style={{ margin: '24px 0' }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>hoặc</Text>
        </Divider>

        {/* Register Link */}
        <div style={{ textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            Chưa có tài khoản?{' '}
            <Link to="/register" style={{ color: '#1890ff', fontWeight: 500 }}>
              Đăng ký ngay
            </Link>
          </Text>
        </div>

        {/* Default Credentials Hint */}
        <div style={{
          marginTop: '24px',
          padding: '12px',
          background: '#f0f0f0',
          borderRadius: '4px',
          textAlign: 'center'
        }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Demo: admin / admin123
          </Text>
        </div>
      </Card>
    </div>
  )
}

export default LoginPage
