/**
 * ProfileCard Component
 * 
 * Displays user profile information with avatar, name, and join date.
 * Includes edit profile button for quick access.
 */
import React from 'react'
import { Card, Avatar, Typography, Space, Button } from 'antd'
import { UserOutlined, EditOutlined, CalendarOutlined } from '@ant-design/icons'
import { useAppSelector } from '../../store/hooks'
import { useTheme } from '../layout/ThemeProvider'

const { Text, Title } = Typography

const ProfileCard = () => {
  const user = useAppSelector((state) => state.auth.user)
  const { darkMode } = useTheme()
  
  const joinDate = user?.created_at 
    ? new Date(user.created_at).toLocaleDateString('vi-VN', { 
        month: 'long', 
        year: 'numeric' 
      })
    : 'Gần đây'

  // Generate beautiful gradient based on username
  const getAvatarGradient = (username) => {
    const gradients = [
      'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
      'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
      'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
      'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
      'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)',
      'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
      'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)',
    ]
    if (!username) return gradients[0]
    const index = username.charCodeAt(0) % gradients.length
    return gradients[index]
  }

  return (
    <Card
      className="profile-card"
      style={{
        borderRadius: '12px',
        border: 'none',
        boxShadow: darkMode ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.08)',
        background: darkMode ? '#252836' : '#ffffff',
        border: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : 'none',
      }}
    >
      <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <Avatar
          size={80}
          style={{
            background: getAvatarGradient(user?.username),
            marginBottom: '12px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            border: '3px solid rgba(255, 255, 255, 0.2)',
          }}
        >
          {user?.username?.charAt(0).toUpperCase() || 'A'}
        </Avatar>
        <Title level={4} style={{ 
          margin: '8px 0 4px', 
          color: darkMode ? '#ffffff' : '#262626' 
        }}>
          {user?.full_name || user?.username || 'Admin'}
        </Title>
        <Space>
          <CalendarOutlined style={{ 
            color: darkMode ? 'rgba(255, 255, 255, 0.7)' : '#8c8c8c' 
          }} />
          <Text style={{ 
            fontSize: '12px',
            color: darkMode ? 'rgba(255, 255, 255, 0.7)' : '#8c8c8c'
          }}>
            Tham gia {joinDate}
          </Text>
        </Space>
      </div>

      <div style={{ marginTop: '20px' }}>
        <Button
          type="primary"
          block
          icon={<EditOutlined />}
          style={{
            borderRadius: '8px',
            height: '40px',
          }}
        >
          Chỉnh sửa hồ sơ
        </Button>
      </div>
    </Card>
  )
}

export default ProfileCard
