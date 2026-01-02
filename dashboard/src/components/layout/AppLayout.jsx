import React, { useState, useEffect } from 'react'
import { Layout, Menu, Avatar, Dropdown, Badge, Switch, Tooltip, Typography } from 'antd'
import {
  DashboardOutlined,
  BarChartOutlined,
  FileTextOutlined,
  SettingOutlined,
  LogoutOutlined,
  MessageOutlined,
  SearchOutlined,
  FilterOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import { logout } from '../../store/slices/authSlice'
import { useTheme } from './ThemeProvider'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

const AppLayout = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const navigate = useNavigate()
  const location = useLocation()
  const dispatch = useAppDispatch()
  const user = useAppSelector((state) => state.auth.user)
  const { darkMode, toggleDarkMode } = useTheme()

  // Real-time clock
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined style={{ fontSize: '18px' }} />,
      label: 'Dashboard',
    },
    {
      key: '/ads',
      icon: <FileTextOutlined style={{ fontSize: '18px' }} />,
      label: 'Bills',
    },
    {
      key: '/analytics',
      icon: <BarChartOutlined style={{ fontSize: '18px' }} />,
      label: 'Analytics',
    },
    {
      key: '/ai-agent',
      icon: <MessageOutlined style={{ fontSize: '18px' }} />,
      label: 'Messages',
    },
    {
      key: '/settings',
      icon: <SettingOutlined style={{ fontSize: '18px' }} />,
      label: 'Settings',
    },
  ]

  const userMenuItems = [
    {
      key: 'profile',
      label: 'Hồ sơ',
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      label: 'Đăng xuất',
      icon: <LogoutOutlined />,
      danger: true,
      onClick: () => {
        dispatch(logout())
        navigate('/login')
      },
    },
  ]

  return (
    <Layout className="min-h-screen">
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        theme="dark"
        width={280}
        style={{
          background: '#ffffff',
          boxShadow: '2px 0 8px rgba(0, 0, 0, 0.08)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Logo/Title */}
        <div className="h-20 flex items-center px-6 border-b" style={{ borderColor: '#e8e8e8' }}>
          {!collapsed && (
            <Title level={3} style={{ margin: 0, color: '#262626', fontWeight: 700, fontSize: '24px' }}>
              Dashboard
            </Title>
          )}
        </div>

        {/* Menu Section */}
        <div style={{ padding: '20px 0' }}>
          {!collapsed && (
            <div style={{ padding: '0 20px', marginBottom: '12px' }}>
              <Text style={{ color: '#8c8c8c', fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Menu
              </Text>
            </div>
          )}
          <Menu
            theme="light"
            mode="inline"
            selectedKeys={[location.pathname === '/' ? '/' : location.pathname]}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
            style={{
              background: 'transparent',
              border: 'none',
            }}
            className="custom-menu"
          />
        </div>

        {/* Scheduled Launches Section */}
        {!collapsed && (
          <div style={{ padding: '20px', borderTop: '1px solid #e8e8e8', marginTop: 'auto' }}>
            <Text style={{ color: '#8c8c8c', fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', display: 'block', marginBottom: '16px' }}>
              Scheduled Launches
            </Text>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {[
                { name: 'Asana Mobile App', color: '#faad14' },
                { name: 'GND Infographics', color: '#ff7875' },
                { name: 'Olympics Website', color: '#13c2c2' },
              ].map((item, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer', padding: '8px', borderRadius: '6px', transition: 'background 0.2s' }} 
                     onMouseEnter={(e) => e.currentTarget.style.background = '#f5f5f5'}
                     onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: item.color, flexShrink: 0 }} />
                  <Text style={{ color: '#262626', fontSize: '14px' }}>{item.name}</Text>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Logout Button */}
        <div style={{ padding: '20px', borderTop: '1px solid #e8e8e8' }}>
          <Menu
            theme="light"
            mode="inline"
            items={[{
              key: 'logout',
              icon: <LogoutOutlined />,
              label: 'Log out',
              danger: true,
              onClick: () => {
                dispatch(logout())
                navigate('/login')
              },
            }]}
            style={{
              background: 'transparent',
              border: 'none',
            }}
          />
        </div>
      </Sider>

      <Layout>
        {/* Header */}
        <Header
          className="bg-white px-6 flex items-center justify-between"
          style={{
            background: '#ffffff',
            borderBottom: '1px solid #e8e8e8',
            boxShadow: '0 1px 4px rgba(0, 0, 0, 0.04)',
            height: '72px',
          }}
        >
          <div className="flex items-center gap-4" style={{ flex: 1 }}>
            {/* Search Bar */}
            <div style={{
              position: 'relative',
              flex: '0 0 300px',
            }}>
              <input
                type="text"
                placeholder="Specify your search"
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 40px 0 16px',
                  border: '1px solid #e8e8e8',
                  borderRadius: '8px',
                  fontSize: '14px',
                  outline: 'none',
                  transition: 'all 0.2s',
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#1890ff'
                  e.target.style.boxShadow = '0 0 0 2px rgba(24, 144, 255, 0.1)'
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = '#e8e8e8'
                  e.target.style.boxShadow = 'none'
                }}
              />
              <SearchOutlined style={{
                position: 'absolute',
                right: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: '#8c8c8c',
                cursor: 'pointer',
                fontSize: '16px',
              }} />
            </div>

            {/* Filter & Favs */}
            <div className="flex items-center gap-3">
              <Tooltip title="Filter">
                <div style={{
                  width: '40px',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'background 0.2s',
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = '#f5f5f5'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                  <FilterOutlined style={{ fontSize: '18px', color: '#595959' }} />
                </div>
              </Tooltip>
              <div className="flex items-center gap-2">
                <Text style={{ fontSize: '14px', color: '#595959' }}>Favs</Text>
                <Switch size="small" />
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Real-time Clock */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-end',
              marginRight: '16px',
            }}>
              <Text style={{ 
                fontSize: '14px', 
                color: '#8c8c8c',
                fontWeight: 500,
                lineHeight: '1.2',
              }}>
                {currentTime.toLocaleDateString('en-US', { 
                  weekday: 'short', 
                  month: 'short', 
                  day: 'numeric',
                  year: 'numeric'
                })}
              </Text>
              <Text style={{ 
                fontSize: '18px', 
                color: '#262626',
                fontWeight: 600,
                lineHeight: '1.2',
                marginTop: '4px',
              }}>
                {currentTime.toLocaleTimeString('en-US', { 
                  hour: '2-digit', 
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: false
                })}
              </Text>
            </div>
            <Title level={4} style={{ margin: 0, color: '#262626', fontWeight: 600 }}>
              My Profile
            </Title>
            <span style={{ fontSize: '20px', color: '#8c8c8c', cursor: 'pointer' }}>⋯</span>
          </div>
        </Header>

        {/* Content */}
        <Content
          className="p-6 min-h-[calc(100vh-64px)]"
          style={{
            background: darkMode 
              ? '#1a1d29' 
              : '#fafafa',
          }}
        >
          <div className="animate-fade-in">{children}</div>
        </Content>
      </Layout>
    </Layout>
  )
}

export default AppLayout
