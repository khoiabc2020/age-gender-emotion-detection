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
                    textDecoration: 'none',
                    fontWeight: 500,
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = '#40a9ff'
                    e.currentTarget.style.textDecoration = 'underline'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = '#1890ff'
                    e.currentTarget.style.textDecoration = 'none'
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
                color: '#262626',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#1890ff'
                e.currentTarget.style.background = '#f0f9ff'
                e.currentTarget.style.transform = 'translateY(-1px)'
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(24, 144, 255, 0.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#e8e8e8'
                e.currentTarget.style.background = '#ffffff'
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <span style={{ fontSize: '18px', fontWeight: 'bold' }}>G</span>
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
                color: '#262626',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#1890ff'
                e.currentTarget.style.background = '#f0f9ff'
                e.currentTarget.style.transform = 'translateY(-1px)'
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(24, 144, 255, 0.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#e8e8e8'
                e.currentTarget.style.background = '#ffffff'
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <span style={{ fontSize: '18px', fontWeight: 'bold' }}>f</span>
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
        background: 'linear-gradient(135deg, #1890ff 0%, #096dd9 50%, #0050b3 100%)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '60px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Animated Background decorations */}
        <div style={{
          position: 'absolute',
          top: '-20%',
          right: '-20%',
          width: '600px',
          height: '600px',
          background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
          borderRadius: '50%',
          animation: 'pulse 8s ease-in-out infinite'
        }} />
        <div style={{
          position: 'absolute',
          bottom: '-15%',
          left: '-15%',
          width: '500px',
          height: '500px',
          background: 'radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%)',
          borderRadius: '50%',
          animation: 'pulse 10s ease-in-out infinite'
        }} />
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '400px',
          height: '400px',
          background: 'radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%)',
          borderRadius: '50%',
          animation: 'pulse 12s ease-in-out infinite'
        }} />

        <div style={{ position: 'relative', zIndex: 1, width: '100%', maxWidth: '520px' }}>
          {/* Welcome Section */}
          <div style={{ marginBottom: '40px' }}>
            <Title level={1} style={{ 
              color: '#ffffff', 
              fontSize: '42px', 
              fontWeight: 700, 
              marginBottom: '16px',
              lineHeight: '1.2',
              textShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
            }}>
              Chào mừng trở lại!
            </Title>
            <Text style={{ 
              color: 'rgba(255, 255, 255, 0.95)', 
              fontSize: '18px',
              lineHeight: '1.6',
              display: 'block',
              textShadow: '0 1px 4px rgba(0, 0, 0, 0.1)'
            }}>
              Vui lòng đăng nhập vào tài khoản Smart Retail của bạn để tiếp tục sử dụng các tính năng phân tích và quản lý.
            </Text>
          </div>

          {/* Preview Cards */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.98)',
            borderRadius: '20px',
            padding: '28px',
            boxShadow: '0 12px 32px rgba(0, 0, 0, 0.2)',
            marginBottom: '20px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.3)'
          }}>
            <div style={{ marginBottom: '24px' }}>
              <Text style={{ 
                fontSize: '18px', 
                fontWeight: 700, 
                color: '#1a1a1a',
                letterSpacing: '0.3px'
              }}>
                Báo cáo Tổng quan
              </Text>
            </div>
            <div style={{
              display: 'flex',
              gap: '16px',
              marginBottom: '24px'
            }}>
              <div style={{
                flex: 1,
                padding: '20px',
                background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                borderRadius: '12px',
                border: '1px solid rgba(24, 144, 255, 0.1)',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)'
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(24, 144, 255, 0.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = 'none'
              }}
              >
                <Text style={{ 
                  fontSize: '13px', 
                  color: '#1890ff', 
                  display: 'block', 
                  marginBottom: '12px',
                  fontWeight: 600,
                  letterSpacing: '0.3px'
                }}>
                  Tổng tương tác
                </Text>
                <Text style={{ 
                  fontSize: '32px', 
                  fontWeight: 700, 
                  color: '#1890ff',
                  lineHeight: '1.2'
                }}>
                  1,234
                </Text>
              </div>
              <div style={{
                flex: 1,
                padding: '20px',
                background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                borderRadius: '12px',
                border: '1px solid rgba(82, 196, 26, 0.1)',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)'
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(82, 196, 26, 0.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = 'none'
              }}
              >
                <Text style={{ 
                  fontSize: '13px', 
                  color: '#52c41a', 
                  display: 'block', 
                  marginBottom: '12px',
                  fontWeight: 600,
                  letterSpacing: '0.3px'
                }}>
                  Khách hàng
                </Text>
                <Text style={{ 
                  fontSize: '32px', 
                  fontWeight: 700, 
                  color: '#52c41a',
                  lineHeight: '1.2'
                }}>
                  567
                </Text>
              </div>
            </div>
            <div style={{
              height: '140px',
              background: 'linear-gradient(135deg, #e6f7ff 0%, #bae7ff 50%, #91d5ff 100%)',
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(24, 144, 255, 0.2)',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%)',
                animation: 'shimmer 3s infinite'
              }} />
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '12px',
                position: 'relative',
                zIndex: 1
              }}>
                <span style={{ fontSize: '32px' }}>📊</span>
                <Text style={{ 
                  color: '#1890ff', 
                  fontSize: '15px',
                  fontWeight: 600,
                  letterSpacing: '0.3px'
                }}>
                  Biểu đồ phân tích dữ liệu
                </Text>
              </div>
            </div>
          </div>

          {/* Classification Card */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.98)',
            borderRadius: '20px',
            padding: '28px',
            boxShadow: '0 12px 32px rgba(0, 0, 0, 0.2)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.3)'
          }}>
            <div style={{ marginBottom: '20px' }}>
              <Text style={{ 
                fontSize: '18px', 
                fontWeight: 700, 
                color: '#1a1a1a',
                letterSpacing: '0.3px'
              }}>
                Phân loại phổ biến
              </Text>
            </div>
            <div style={{
              display: 'flex',
              gap: '16px',
              flexWrap: 'wrap'
            }}>
              {[
                { name: 'Nam', color: '#1890ff', gradient: 'linear-gradient(135deg, #e6f7ff 0%, #bae7ff 100%)' },
                { name: 'Nữ', color: '#52c41a', gradient: 'linear-gradient(135deg, #f6ffed 0%, #d9f7be 100%)' },
                { name: 'Trẻ em', color: '#faad14', gradient: 'linear-gradient(135deg, #fffbe6 0%, #ffe58f 100%)' }
              ].map((item, idx) => (
                <div 
                  key={idx} 
                  style={{
                    padding: '20px',
                    background: item.gradient,
                    borderRadius: '12px',
                    flex: '1 1 calc(33% - 11px)',
                    minWidth: '100px',
                    border: `1px solid rgba(${item.color === '#1890ff' ? '24, 144, 255' : item.color === '#52c41a' ? '82, 196, 26' : '250, 173, 20'}, 0.2)`,
                    transition: 'all 0.3s ease',
                    cursor: 'pointer'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-4px) scale(1.02)'
                    e.currentTarget.style.boxShadow = `0 8px 16px rgba(${item.color === '#1890ff' ? '24, 144, 255' : item.color === '#52c41a' ? '82, 196, 26' : '250, 173, 20'}, 0.2)`
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0) scale(1)'
                    e.currentTarget.style.boxShadow = 'none'
                  }}
                >
                  <div style={{
                    width: '48px',
                    height: '48px',
                    background: item.color,
                    borderRadius: '50%',
                    marginBottom: '12px',
                    boxShadow: `0 4px 12px rgba(${item.color === '#1890ff' ? '24, 144, 255' : item.color === '#52c41a' ? '82, 196, 26' : '250, 173, 20'}, 0.3)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '20px',
                    color: '#ffffff',
                    fontWeight: 'bold'
                  }}>
                    {item.name === 'Nam' ? '👨' : item.name === 'Nữ' ? '👩' : '👶'}
                  </div>
                  <Text style={{ 
                    fontSize: '14px', 
                    color: '#1a1a1a', 
                    fontWeight: 600,
                    letterSpacing: '0.3px'
                  }}>
                    {item.name}
                  </Text>
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
