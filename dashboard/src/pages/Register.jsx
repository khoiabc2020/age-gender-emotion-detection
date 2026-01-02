import React, { useState } from 'react'
import { Form, Input, Button, Card, message, Typography, Divider } from 'antd'
import { UserOutlined, LockOutlined, MailOutlined, ArrowLeftOutlined } from '@ant-design/icons'
import { useNavigate, Link } from 'react-router-dom'
import api from '../services/api'

const { Title, Text } = Typography

const RegisterPage = () => {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const onFinish = async (values) => {
    if (values.password !== values.confirmPassword) {
      message.error('Mật khẩu xác nhận không khớp!')
      return
    }

    setLoading(true)
    try {
      const response = await api.post('/api/v1/auth/register', {
        username: values.username,
        email: values.email,
        password: values.password,
        full_name: values.full_name || values.username
      })
      
      if (response.data) {
        message.success('Đăng ký thành công! Vui lòng đăng nhập.')
        navigate('/login')
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Đăng ký thất bại!'
      message.error(errorMessage)
      console.error('Register error:', error)
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
            Đăng ký tài khoản
          </Title>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            Tạo tài khoản mới để bắt đầu
          </Text>
        </div>

        {/* Register Form */}
        <Form
          name="register"
          onFinish={onFinish}
          autoComplete="off"
          size="large"
          layout="vertical"
        >
          <Form.Item
            name="username"
            rules={[
              { required: true, message: 'Vui lòng nhập tên đăng nhập!' },
              { min: 3, message: 'Tên đăng nhập phải có ít nhất 3 ký tự!' },
            ]}
          >
            <Input
              prefix={<UserOutlined style={{ color: '#bfbfbf' }} />}
              placeholder="Tên đăng nhập"
              style={{ borderRadius: '4px' }}
            />
          </Form.Item>

          <Form.Item
            name="email"
            rules={[
              { required: true, message: 'Vui lòng nhập email!' },
              { type: 'email', message: 'Email không hợp lệ!' },
            ]}
          >
            <Input
              prefix={<MailOutlined style={{ color: '#bfbfbf' }} />}
              placeholder="Email"
              style={{ borderRadius: '4px' }}
            />
          </Form.Item>

          <Form.Item
            name="full_name"
          >
            <Input
              prefix={<UserOutlined style={{ color: '#bfbfbf' }} />}
              placeholder="Họ và tên (tùy chọn)"
              style={{ borderRadius: '4px' }}
            />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[
              { required: true, message: 'Vui lòng nhập mật khẩu!' },
              { min: 6, message: 'Mật khẩu phải có ít nhất 6 ký tự!' },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
              placeholder="Mật khẩu"
              style={{ borderRadius: '4px' }}
            />
          </Form.Item>

          <Form.Item
            name="confirmPassword"
            rules={[
              { required: true, message: 'Vui lòng xác nhận mật khẩu!' },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
              placeholder="Xác nhận mật khẩu"
              style={{ borderRadius: '4px' }}
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
              Đăng ký
            </Button>
          </Form.Item>
        </Form>

        <Divider style={{ margin: '24px 0' }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>hoặc</Text>
        </Divider>

        {/* Login Link */}
        <div style={{ textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            Đã có tài khoản?{' '}
            <Link to="/login" style={{ color: '#1890ff', fontWeight: 500 }}>
              Đăng nhập
            </Link>
          </Text>
        </div>
      </Card>
    </div>
  )
}

export default RegisterPage
