import React, { useState } from 'react'
import { Form, Input, Button, Card, message } from 'antd'
import { UserOutlined, LockOutlined, DashboardOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { useAppDispatch } from '../store/hooks'
import { login } from '../store/slices/authSlice'
import api from '../services/api'

const LoginPage = () => {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const dispatch = useAppDispatch()

  const onFinish = async (values) => {
    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('username', values.username)
      formData.append('password', values.password)
      
      const response = await api.post('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      })
      
      if (response.data.access_token) {
        dispatch(login({
          token: response.data.access_token,
          user: response.data.user,
        }))
        message.success('Đăng nhập thành công!')
        navigate('/')
      }
    } catch (error) {
      message.error('Tên đăng nhập hoặc mật khẩu không đúng!')
      console.error('Login error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-animated relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-primary rounded-full blur-3xl opacity-30 animate-pulse-slow"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-secondary rounded-full blur-3xl opacity-30 animate-pulse-slow" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-info rounded-full blur-3xl opacity-20 animate-pulse-slow" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10 w-full max-w-md px-4 animate-fade-in">
        <Card
          className="glass backdrop-blur-xl border-0 shadow-2xl"
          style={{
            borderRadius: '24px',
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(20px)',
          }}
        >
          {/* Logo & Title */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-primary rounded-2xl mb-4 shadow-glow">
              <DashboardOutlined className="text-4xl text-white" />
            </div>
            <h1 className="text-3xl font-bold gradient-text mb-2">
              Smart Retail Analytics
            </h1>
            <p className="text-gray-500 text-sm">
              Đăng nhập để tiếp tục
            </p>
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
                prefix={<UserOutlined className="text-gray-400" />}
                placeholder="Tên đăng nhập"
                className="rounded-lg h-12"
                style={{ borderRadius: '12px' }}
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[
                { required: true, message: 'Vui lòng nhập mật khẩu!' },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined className="text-gray-400" />}
                placeholder="Mật khẩu"
                className="rounded-lg h-12"
                style={{ borderRadius: '12px' }}
              />
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                className="w-full h-12 rounded-lg bg-gradient-primary border-0 font-semibold text-base shadow-lg hover:shadow-xl transition-all duration-300 btn-gradient"
                style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                }}
              >
                Đăng nhập
              </Button>
            </Form.Item>
          </Form>

          {/* Footer Info */}
          <div className="text-center mt-6 text-sm text-gray-500">
            <p>Mặc định: <span className="font-semibold">admin / admin123</span></p>
          </div>
        </Card>

        {/* Decorative Elements */}
        <div className="absolute -z-10 top-1/4 left-1/4 w-32 h-32 bg-gradient-primary rounded-full blur-2xl opacity-20 animate-pulse"></div>
        <div className="absolute -z-10 bottom-1/4 right-1/4 w-40 h-40 bg-gradient-secondary rounded-full blur-2xl opacity-20 animate-pulse" style={{ animationDelay: '1.5s' }}></div>
      </div>
    </div>
  )
}

export default LoginPage
