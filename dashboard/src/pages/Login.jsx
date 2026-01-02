import React, { useState, useEffect } from 'react'
import { Form, Input, Button, Checkbox, message, Typography, Divider } from 'antd'
import { UserOutlined, LockOutlined } from '@ant-design/icons'
import { useNavigate, Link } from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../store/hooks'
import { login as loginThunk } from '../store/slices/authSlice'

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
      if (process.env.NODE_ENV === 'development') {
        console.error('Login error:', error)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      background: '#ffffff'
    }}>
      {/* Left Section - Login Form */}
      <div style={{
        flex: '0 0 50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px',
        background: '#ffffff'
      }}>
        <div style={{
          width: '100%',
          maxWidth: '440px'
        }}>
          {/* Logo/Title */}
          <div style={{ marginBottom: '48px' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              marginBottom: '48px'
            }}>
              <div style={{
                width: '48px',
                height: '48px',
                background: 'linear-gradient(135deg, #1890ff 0%, #096dd9 100%)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(24, 144, 255, 0.3)'
              }}>
                <span style={{ fontSize: '24px', color: '#ffffff', fontWeight: 'bold' }}>S</span>
              </div>
              <Title level={2} style={{ margin: 0, color: '#262626', fontWeight: 700, fontSize: '24px' }}>
                Smart Retail
              </Title>
            </div>
            <Title level={1} style={{ marginBottom: '8px', color: '#262626', fontWeight: 700, fontSize: '32px' }}>
              Đăng nhập
            </Title>
            <Text style={{ fontSize: '16px', color: '#8c8c8c' }}>
              Chào mừng trở lại! Vui lòng nhập thông tin đăng nhập của bạn
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
              label={<span style={{ fontSize: '14px', fontWeight: 500, color: '#262626' }}>Tên đăng nhập</span>}
              rules={[
                { required: true, message: 'Vui lòng nhập tên đăng nhập!' },
              ]}
            >
              <Input
                prefix={<UserOutlined style={{ color: '#bfbfbf' }} />}
                placeholder="Nhập tên đăng nhập"
                style={{ 
                  borderRadius: '8px',
                  height: '48px',
                  fontSize: '15px'
                }}
              />
            </Form.Item>

            <Form.Item
              name="password"
              label={<span style={{ fontSize: '14px', fontWeight: 500, color: '#262626' }}>Mật khẩu</span>}
              rules={[
                { required: true, message: 'Vui lòng nhập mật khẩu!' },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
                placeholder="Nhập mật khẩu"
                style={{ 
                  borderRadius: '8px',
                  height: '48px',
                  fontSize: '15px'
                }}
              />
            </Form.Item>

            <Form.Item style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Form.Item name="remember" valuePropName="checked" noStyle>
                  <Checkbox style={{ fontSize: '14px', color: '#595959' }}>Ghi nhớ đăng nhập 30 ngày</Checkbox>
                </Form.Item>
                <Link 
                  to="#" 
                  style={{ 
                    color: '#1890ff', 
                    fontSize: '14px',
                    textDecoration: 'none'
                  }}
                >
                  Quên mật khẩu?
                </Link>
              </div>
            </Form.Item>

            <Form.Item style={{ marginBottom: '24px' }}>
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
                  background: '#1890ff',
                  border: 'none',
                  boxShadow: '0 2px 8px rgba(24, 144, 255, 0.3)',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#40a9ff'
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(24, 144, 255, 0.4)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = '#1890ff'
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(24, 144, 255, 0.3)'
                }}
              >
                Đăng nhập
              </Button>
            </Form.Item>
          </Form>

          <Divider style={{ margin: '32px 0' }}>
            <Text style={{ fontSize: '13px', color: '#bfbfbf', fontWeight: 500 }}>HOẶC</Text>
          </Divider>

          {/* Social Login Buttons */}
          <div style={{ display: 'flex', gap: '12px', marginBottom: '32px' }}>
            <Button
              block
              size="large"
              style={{
                height: '48px',
                borderRadius: '8px',
                border: '1px solid #e8e8e8',
                background: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                fontSize: '14px',
                fontWeight: 500,
                color: '#262626'
              }}
            >
              <span style={{ fontSize: '18px' }}>G</span>
              Đăng nhập với Google
            </Button>
            <Button
              block
              size="large"
              style={{
                height: '48px',
                borderRadius: '8px',
                border: '1px solid #e8e8e8',
                background: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                fontSize: '14px',
                fontWeight: 500,
                color: '#262626'
              }}
            >
              <span style={{ fontSize: '18px' }}>f</span>
              Đăng nhập với Facebook
            </Button>
          </div>

          {/* Register Link */}
          <div style={{ textAlign: 'center' }}>
            <Text style={{ fontSize: '14px', color: '#8c8c8c' }}>
              Chưa có tài khoản?{' '}
              <Link 
                to="/register" 
                style={{ 
                  color: '#1890ff', 
                  fontWeight: 600,
                  textDecoration: 'none'
                }}
              >
                Đăng ký ngay
              </Link>
            </Text>
          </div>
        </div>
      </div>

      {/* Right Section - Preview/Dashboard */}
      <div style={{
        flex: '0 0 50%',
        background: 'linear-gradient(135deg, #1890ff 0%, #096dd9 100%)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '60px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Background decoration */}
        <div style={{
          position: 'absolute',
          top: '-20%',
          right: '-20%',
          width: '500px',
          height: '500px',
          background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
          borderRadius: '50%'
        }} />
        <div style={{
          position: 'absolute',
          bottom: '-10%',
          left: '-10%',
          width: '400px',
          height: '400px',
          background: 'radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%)',
          borderRadius: '50%'
        }} />

        <div style={{ position: 'relative', zIndex: 1, width: '100%', maxWidth: '500px' }}>
          <Title level={1} style={{ 
            color: '#ffffff', 
            fontSize: '36px', 
            fontWeight: 700, 
            marginBottom: '16px',
            lineHeight: '1.3'
          }}>
            Chào mừng trở lại!
          </Title>
          <Text style={{ 
            color: 'rgba(255, 255, 255, 0.9)', 
            fontSize: '18px',
            lineHeight: '1.6',
            display: 'block',
            marginBottom: '48px'
          }}>
            Vui lòng đăng nhập vào tài khoản Smart Retail của bạn để tiếp tục sử dụng các tính năng phân tích và quản lý.
          </Text>

          {/* Preview Cards */}
          <div style={{
            background: '#ffffff',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
            marginBottom: '20px'
          }}>
            <div style={{ marginBottom: '20px' }}>
              <Text style={{ fontSize: '16px', fontWeight: 600, color: '#262626' }}>Báo cáo Tổng quan</Text>
            </div>
            <div style={{
              display: 'flex',
              gap: '16px',
              marginBottom: '20px'
            }}>
              <div style={{
                flex: 1,
                padding: '16px',
                background: '#f5f5f5',
                borderRadius: '8px'
              }}>
                <Text style={{ fontSize: '12px', color: '#8c8c8c', display: 'block', marginBottom: '8px' }}>Tổng tương tác</Text>
                <Text style={{ fontSize: '24px', fontWeight: 700, color: '#262626' }}>1,234</Text>
              </div>
              <div style={{
                flex: 1,
                padding: '16px',
                background: '#f5f5f5',
                borderRadius: '8px'
              }}>
                <Text style={{ fontSize: '12px', color: '#8c8c8c', display: 'block', marginBottom: '8px' }}>Khách hàng</Text>
                <Text style={{ fontSize: '24px', fontWeight: 700, color: '#262626' }}>567</Text>
              </div>
            </div>
            <div style={{
              height: '120px',
              background: 'linear-gradient(135deg, #e6f7ff 0%, #bae7ff 100%)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Text style={{ color: '#1890ff', fontSize: '14px' }}>📊 Biểu đồ phân tích dữ liệu</Text>
            </div>
          </div>

          <div style={{
            background: '#ffffff',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)'
          }}>
            <div style={{ marginBottom: '16px' }}>
              <Text style={{ fontSize: '16px', fontWeight: 600, color: '#262626' }}>Phân loại phổ biến</Text>
            </div>
            <div style={{
              display: 'flex',
              gap: '12px',
              flexWrap: 'wrap'
            }}>
              {['Nam', 'Nữ', 'Trẻ em'].map((item, idx) => (
                <div key={idx} style={{
                  padding: '12px 16px',
                  background: '#f5f5f5',
                  borderRadius: '8px',
                  flex: '1 1 calc(33% - 8px)',
                  minWidth: '80px'
                }}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    background: ['#1890ff', '#52c41a', '#faad14'][idx],
                    borderRadius: '50%',
                    marginBottom: '8px'
                  }} />
                  <Text style={{ fontSize: '12px', color: '#262626', fontWeight: 500 }}>{item}</Text>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LoginPage
