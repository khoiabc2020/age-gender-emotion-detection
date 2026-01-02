/**
 * QuickActions Component
 * 
 * Provides quick access buttons for common actions:
 * Reports, Messages, Notifications, and Settings.
 */
import React from 'react'
import { Card, Space, Button } from 'antd'
import {
  FileTextOutlined,
  MessageOutlined,
  BellOutlined,
  SettingOutlined,
} from '@ant-design/icons'

const QuickActions = () => {
  const actions = [
    { icon: <FileTextOutlined />, label: 'Báo cáo', color: '#1890ff' },
    { icon: <MessageOutlined />, label: 'Tin nhắn', color: '#52c41a' },
    { icon: <BellOutlined />, label: 'Thông báo', color: '#faad14' },
    { icon: <SettingOutlined />, label: 'Cài đặt', color: '#722ed1' },
  ]

  return (
    <Card
      className="quick-actions-card"
      style={{
        borderRadius: '12px',
        border: 'none',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        background: 'var(--bg-primary)',
      }}
      bodyStyle={{ padding: '16px' }}
    >
      <Space size="large" style={{ width: '100%', justifyContent: 'space-around' }}>
        {actions.map((action, index) => (
          <Button
            key={index}
            type="text"
            shape="circle"
            size="large"
            icon={action.icon}
            style={{
              width: '56px',
              height: '56px',
              fontSize: '24px',
              color: action.color,
              background: `${action.color}15`,
              border: 'none',
            }}
            title={action.label}
          />
        ))}
      </Space>
    </Card>
  )
}

export default QuickActions
