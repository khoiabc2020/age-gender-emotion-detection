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
          background: '#001529',
          boxShadow: '2px 0 8px rgba(0, 0, 0, 0.1)',
        }}
      >
        {/* Logo */}
        <div className="h-16 flex items-center justify-center border-b border-gray-700" style={{ borderColor: 'rgba(255, 255, 255, 0.1)' }}>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
              <DashboardOutlined className="text-white text-lg" />
            </div>
            {!collapsed && (
              <span className="text-white text-lg font-semibold">
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
          className="bg-white px-6 flex items-center justify-between"
          style={{
            background: '#ffffff',
            borderBottom: '1px solid #e8e8e8',
            boxShadow: '0 1px 4px rgba(0, 0, 0, 0.04)',
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
                  style={{
                    background: '#1890ff',
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
            background: darkMode ? '#1a1a1a' : '#fafafa',
          }}
        >
          <div className="animate-fade-in">{children}</div>
        </Content>
      </Layout>
    </Layout>
  )
}

export default AppLayout
