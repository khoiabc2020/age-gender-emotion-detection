import React, { useState } from 'react'
import { Card, Form, Input, Button, Switch, Select, Divider, Typography, message, Space } from 'antd'
import { SaveOutlined, UserOutlined, BellOutlined, SecurityScanOutlined, PaletteOutlined, RobotOutlined, KeyOutlined } from '@ant-design/icons'
import { useAppSelector } from '../store/hooks'

const { Title, Text } = Typography
const { Option } = Select

const SettingsPage = () => {
  const [form] = Form.useForm()
  const user = useAppSelector((state) => state.auth.user)
  const [loading, setLoading] = useState(false)

  const onFinish = async (values) => {
    setLoading(true)
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000))
      message.success('Cài đặt đã được lưu thành công!')
    } catch (error) {
      message.error('Lỗi khi lưu cài đặt')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="animate-fade-in max-w-4xl">
      <Title level={2} className="gradient-text mb-6">
        Cài đặt
      </Title>

      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
        initialValues={{
          notifications: true,
          email_notifications: true,
          theme: 'light',
          language: 'vi',
        }}
      >
        {/* Profile Settings */}
        <Card
          className="mb-6 card-hover border-0 shadow-lg"
          style={{ borderRadius: '20px', background: 'rgba(255, 255, 255, 0.95)' }}
          title={
            <Space>
              <UserOutlined className="text-lg" style={{ color: '#667eea' }} />
              <span className="text-lg font-semibold">Thông tin cá nhân</span>
            </Space>
          }
        >
          <Form.Item label="Tên đăng nhập" name="username">
            <Input
              value={user?.username || 'admin'}
              disabled
              className="rounded-lg"
            />
          </Form.Item>
          <Form.Item label="Email" name="email">
            <Input
              value={user?.email || 'admin@retail.com'}
              disabled
              className="rounded-lg"
            />
          </Form.Item>
          <Form.Item label="Tên hiển thị" name="display_name">
            <Input placeholder="Nhập tên hiển thị" className="rounded-lg" />
          </Form.Item>
        </Card>

        {/* Notification Settings */}
        <Card
          className="mb-6 card-hover border-0 shadow-lg"
          style={{ borderRadius: '20px', background: 'rgba(255, 255, 255, 0.95)' }}
          title={
            <Space>
              <BellOutlined className="text-lg" style={{ color: '#667eea' }} />
              <span className="text-lg font-semibold">Thông báo</span>
            </Space>
          }
        >
          <Form.Item
            name="notifications"
            valuePropName="checked"
            label="Bật thông báo"
          >
            <Switch />
          </Form.Item>
          <Form.Item
            name="email_notifications"
            valuePropName="checked"
            label="Thông báo qua Email"
          >
            <Switch />
          </Form.Item>
          <Form.Item
            name="push_notifications"
            valuePropName="checked"
            label="Thông báo đẩy"
          >
            <Switch />
          </Form.Item>
        </Card>

        {/* Appearance Settings */}
        <Card
          className="mb-6 card-hover border-0 shadow-lg"
          style={{ borderRadius: '20px', background: 'rgba(255, 255, 255, 0.95)' }}
          title={
            <Space>
              <PaletteOutlined className="text-lg" style={{ color: '#667eea' }} />
              <span className="text-lg font-semibold">Giao diện</span>
            </Space>
          }
        >
          <Form.Item label="Chủ đề" name="theme">
            <Select className="rounded-lg">
              <Option value="light">Sáng</Option>
              <Option value="dark">Tối</Option>
              <Option value="auto">Tự động</Option>
            </Select>
          </Form.Item>
          <Form.Item label="Ngôn ngữ" name="language">
            <Select className="rounded-lg">
              <Option value="vi">Tiếng Việt</Option>
              <Option value="en">English</Option>
            </Select>
          </Form.Item>
        </Card>

        {/* AI Agent Settings */}
        <Card
          className="mb-6 card-hover border-0 shadow-lg"
          style={{ borderRadius: '20px', background: 'rgba(255, 255, 255, 0.95)' }}
          title={
            <Space>
              <RobotOutlined className="text-lg" style={{ color: '#667eea' }} />
              <span className="text-lg font-semibold">AI Agent Configuration</span>
            </Space>
          }
        >
          <div className="mb-4 p-3 rounded-lg bg-blue-50 border border-blue-200">
            <Text className="text-sm text-blue-700">
              <KeyOutlined className="mr-2" />
              Cấu hình API keys để sử dụng AI Agent. Hỗ trợ Google AI (Gemini) và ChatGPT.
            </Text>
          </div>
          <Form.Item
            label="Google AI API Key"
            name="google_ai_api_key"
            help="Lấy API key từ https://makersuite.google.com/app/apikey"
          >
            <Input.Password
              placeholder="Nhập Google AI API key"
              className="rounded-lg"
            />
          </Form.Item>
          <Form.Item
            label="OpenAI API Key (ChatGPT)"
            name="openai_api_key"
            help="Lấy API key từ https://platform.openai.com/api-keys"
          >
            <Input.Password
              placeholder="Nhập OpenAI API key"
              className="rounded-lg"
            />
          </Form.Item>
          <Form.Item
            label="AI Provider"
            name="ai_provider"
            help="Chọn provider mặc định (có thể dùng cả hai)"
          >
            <Select className="rounded-lg">
              <Option value="google_ai">Google AI (Gemini)</Option>
              <Option value="chatgpt">ChatGPT</Option>
              <Option value="both">Cả hai (Kết hợp kết quả)</Option>
            </Select>
          </Form.Item>
        </Card>

        {/* Security Settings */}
        <Card
          className="mb-6 card-hover border-0 shadow-lg"
          style={{ borderRadius: '20px', background: 'rgba(255, 255, 255, 0.95)' }}
          title={
            <Space>
              <SecurityScanOutlined className="text-lg" style={{ color: '#667eea' }} />
              <span className="text-lg font-semibold">Bảo mật</span>
            </Space>
          }
        >
          <Form.Item label="Mật khẩu hiện tại" name="current_password">
            <Input.Password placeholder="Nhập mật khẩu hiện tại" className="rounded-lg" />
          </Form.Item>
          <Form.Item label="Mật khẩu mới" name="new_password">
            <Input.Password placeholder="Nhập mật khẩu mới" className="rounded-lg" />
          </Form.Item>
          <Form.Item label="Xác nhận mật khẩu mới" name="confirm_password">
            <Input.Password placeholder="Nhập lại mật khẩu mới" className="rounded-lg" />
          </Form.Item>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end">
          <Button
            type="primary"
            htmlType="submit"
            loading={loading}
            icon={<SaveOutlined />}
            size="large"
            className="rounded-lg bg-gradient-primary border-0 shadow-lg hover:shadow-xl transition-all duration-300"
            style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              height: '48px',
              paddingLeft: '32px',
              paddingRight: '32px',
              fontSize: '16px',
              fontWeight: 600,
            }}
          >
            Lưu cài đặt
          </Button>
        </div>
      </Form>
    </div>
  )
}

export default SettingsPage
