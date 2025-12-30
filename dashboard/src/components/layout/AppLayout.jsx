import React, { useState, useEffect } from 'react'
import { Layout, Menu, Avatar, Dropdown, Badge, Switch, Tooltip } from 'antd'
import {
  DashboardOutlined,
  BarChartOutlined,
  VideoCameraOutlined,
  SettingOutlined,
  LogoutOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MoonOutlined,
  SunOutlined,
  RobotOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import { logout } from '../../store/slices/authSlice'
import { useTheme } from './ThemeProvider'

const { Header, Sider, Content } = Layout

const AppLayout = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const dispatch = useAppDispatch()
  const user = useAppSelector((state) => state.auth.user)
  const { darkMode, toggleDarkMode } = useTheme()

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Tổng quan',
    },
    {
      key: '/analytics',
      icon: <BarChartOutlined />,
      label: 'Phân tích',
    },
    {
      key: '/ads',
      icon: <VideoCameraOutlined />,
      label: 'Quản lý Quảng cáo',
    },
    {
      key: '/ai-agent',
      icon: <RobotOutlined />,
      label: 'AI Agent',
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: 'Cài đặt',
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
        width={260}
        style={{
          background: 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)',
          boxShadow: '4px 0 12px rgba(0, 0, 0, 0.15)',
        }}
      >
        {/* Logo */}
        <div className="h-20 flex items-center justify-center border-b border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-primary rounded-xl flex items-center justify-center shadow-glow">
              <DashboardOutlined className="text-white text-lg" />
            </div>
            {!collapsed && (
              <span className="text-white text-xl font-bold gradient-text-white">
                Smart Retail
              </span>
            )}
          </div>
        </div>

        {/* Menu */}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{
            background: 'transparent',
            border: 'none',
            marginTop: '16px',
          }}
          className="custom-menu"
        />
      </Sider>

      <Layout>
        {/* Header */}
        <Header
          className="bg-white px-6 flex items-center justify-between shadow-sm border-b border-gray-100"
          style={{
            background: 'linear-gradient(90deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%)',
            backdropFilter: 'blur(10px)',
          }}
        >
          <div className="flex items-center gap-4">
            {React.createElement(
              collapsed ? MenuUnfoldOutlined : MenuFoldOutlined,
              {
                className: 'trigger text-xl text-gray-600 hover:text-gray-900 cursor-pointer transition-colors',
                onClick: () => setCollapsed(!collapsed),
              }
            )}
          </div>

          <div className="flex items-center gap-6">
            {/* Dark Mode Toggle */}
            <Tooltip title={darkMode ? 'Chuyển sang sáng' : 'Chuyển sang tối'}>
              <div className="flex items-center gap-2">
                <SunOutlined className="text-gray-400" />
                <Switch
                  checked={darkMode}
                  onChange={toggleDarkMode}
                  checkedChildren={<MoonOutlined />}
                  unCheckedChildren={<SunOutlined />}
                  style={{
                    background: darkMode ? '#667eea' : '#ccc',
                  }}
                />
              </div>
            </Tooltip>

            {/* Notifications */}
            <Badge count={5} size="small">
              <BellOutlined className="text-xl text-gray-600 hover:text-gray-900 cursor-pointer transition-colors" />
            </Badge>

            {/* User Menu */}
            <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
              <div className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity">
                <Avatar
                  className="shadow-md"
                  style={{
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    border: '2px solid white',
                  }}
                  size="large"
                >
                  {user?.username?.charAt(0).toUpperCase() || 'A'}
                </Avatar>
                {!collapsed && (
                  <div className="hidden md:block">
                    <div className="text-sm font-semibold text-gray-700">
                      {user?.full_name || user?.username || 'Admin'}
                    </div>
                    <div className="text-xs text-gray-500">
                      {user?.email || 'admin@retail.com'}
                    </div>
                  </div>
                )}
              </div>
            </Dropdown>
          </div>
        </Header>

        {/* Content */}
        <Content
          className="p-6 min-h-[calc(100vh-64px)]"
          style={{
            background: darkMode
              ? 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)'
              : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
          }}
        >
          <div className="animate-fade-in">{children}</div>
        </Content>
      </Layout>
    </Layout>
  )
}

export default AppLayout
