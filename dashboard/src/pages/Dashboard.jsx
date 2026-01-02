import React, { useEffect } from 'react'
import { Row, Col, Card, Spin, Typography, Space, Divider } from 'antd'
import {
  UserOutlined,
  TeamOutlined,
  SmileOutlined,
  VideoCameraOutlined,
  DollarOutlined,
} from '@ant-design/icons'
import { useAppDispatch, useAppSelector } from '../store/hooks'
import { fetchStats, fetchAgeByHour, fetchEmotionDistribution } from '../store/slices/analyticsSlice'
import AgeChart from '../components/charts/AgeChart'
import EmotionPieChart from '../components/charts/EmotionPieChart'
import GenderChart from '../components/charts/GenderChart'
import AdPerformanceChart from '../components/charts/AdPerformanceChart'
import StatsCard from '../components/dashboard/StatsCard'
import ProfileCard from '../components/dashboard/ProfileCard'
import QuickActions from '../components/dashboard/QuickActions'
import WeeklyInvoice from '../components/dashboard/WeeklyInvoice'

const { Title, Text } = Typography

const DashboardPage = () => {
  const dispatch = useAppDispatch()
  const { stats, ageByHour, emotionDistribution, loading } = useAppSelector(
    (state) => state.analytics
  )
  const user = useAppSelector((state) => state.auth.user)

  useEffect(() => {
    // Initial load with delay to prevent blocking
    const initialTimeout = setTimeout(() => {
      dispatch(fetchStats(24))
      dispatch(fetchAgeByHour(24))
      dispatch(fetchEmotionDistribution(24))
    }, 100)
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      dispatch(fetchStats(24))
      dispatch(fetchAgeByHour(24))
      dispatch(fetchEmotionDistribution(24))
    }, 30000)
    
    return () => {
      clearTimeout(initialTimeout)
      clearInterval(interval)
    }
  }, [dispatch])

  if (loading && !stats) {
    return (
      <div className="flex justify-center items-center h-64">
        <Spin size="large" />
      </div>
    )
  }

  const totalInteractions = stats?.total_interactions || 0
  const uniqueCustomers = stats?.unique_customers || 0
  const avgAge = stats?.avg_age || 0
  const totalAds = stats?.top_ads?.length || 0
  const dailyLimit = 1000
  const usedToday = totalInteractions
  const spentPercentage = Math.min((usedToday / dailyLimit) * 100, 100)

  return (
    <div className="animate-fade-in" style={{ padding: '24px', background: '#fafafa', minHeight: '100%' }}>

      {/* Main Content Grid */}
      <Row gutter={[24, 24]}>
        {/* Left Column - Main Stats */}
        <Col xs={24} lg={16}>
          <Row gutter={[16, 16]}>
            {/* Greeting Card */}
            <Col xs={24} md={12}>
              <Card
                style={{
                  borderRadius: '16px',
                  border: 'none',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  background: '#ffffff',
                  height: '100%',
                }}
                bodyStyle={{ padding: '32px' }}
              >
                <Text style={{ color: '#8c8c8c', fontSize: '14px', display: 'block', marginBottom: '12px' }}>
                  Hello, {user?.full_name || user?.username || 'User'}
                </Text>
                <Title 
                  level={1} 
                  style={{ 
                    margin: 0, 
                    color: '#ff4d4f', 
                    fontSize: '48px',
                    fontWeight: 700,
                    lineHeight: '1.2',
                  }}
                >
                  ${totalInteractions.toLocaleString()}
                </Title>
                <Space style={{ marginTop: '12px' }}>
                  <Text style={{ color: '#8c8c8c', fontSize: '13px' }}>
                    0.00% (US$0.00) last 24 hours
                  </Text>
                  <span style={{ color: '#52c41a', fontSize: '12px' }}>↑</span>
                </Space>
              </Card>
            </Col>

            {/* Spending Progress Card */}
            <Col xs={24} md={12}>
              <Card
                style={{
                  borderRadius: '16px',
                  border: 'none',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  background: 'linear-gradient(135deg, #ff6b6b 0%, #ffa500 50%, #4ecdc4 100%)',
                  height: '100%',
                  position: 'relative',
                  overflow: 'hidden',
                }}
                bodyStyle={{ padding: '32px', textAlign: 'center', position: 'relative', zIndex: 1 }}
              >
                <div style={{
                  position: 'absolute',
                  top: '-50%',
                  right: '-50%',
                  width: '200%',
                  height: '200%',
                  background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
                  zIndex: 0,
                }} />
                <div style={{ marginBottom: '20px', position: 'relative', zIndex: 1 }}>
                  <div style={{
                    width: '140px',
                    height: '140px',
                    margin: '0 auto',
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <div style={{
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      borderRadius: '50%',
                      border: '8px solid rgba(255,255,255,0.3)',
                    }} />
                    <div style={{
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      borderRadius: '50%',
                      border: '8px solid transparent',
                      borderTopColor: '#fff',
                      transform: `rotate(${(spentPercentage / 100) * 360 - 90}deg)`,
                      transition: 'transform 0.3s',
                    }} />
                    <div style={{
                      position: 'absolute',
                      textAlign: 'center',
                    }}>
                      <Text style={{ 
                        color: '#fff', 
                        fontSize: '32px', 
                        fontWeight: 700,
                        display: 'block',
                      }}>
                        {spentPercentage.toFixed(0)}%
                      </Text>
                      <Text style={{ 
                        color: 'rgba(255,255,255,0.95)', 
                        fontSize: '14px',
                        fontWeight: 600,
                        display: 'block',
                      }}>
                        Spent
                      </Text>
                    </div>
                  </div>
                </div>
                <div style={{ position: 'relative', zIndex: 1 }}>
                  <Text style={{ color: 'rgba(255,255,255,0.95)', fontSize: '13px', display: 'block', marginBottom: '6px', fontWeight: 500 }}>
                    Daily Limit ${dailyLimit.toLocaleString()}
                  </Text>
                  <Text style={{ color: '#fff', fontSize: '18px', fontWeight: 700, display: 'block' }}>
                    Used Today ${usedToday.toLocaleString()}
                  </Text>
                </div>
              </Card>
            </Col>

            {/* Today's Stats Section */}
            <Col xs={24}>
              <Card
                style={{
                  borderRadius: '16px',
                  border: 'none',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  background: '#ffffff',
                }}
                bodyStyle={{ padding: '24px' }}
              >
                <div style={{ marginBottom: '16px' }}>
                  <Text style={{ fontSize: '16px', fontWeight: 600, color: '#262626' }}>
                    Today's Stats
                  </Text>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                  {['Mo 12', 'Tu 13', 'We 14', 'Th 15', 'Fr 16', 'Sa 17', 'Su 18', 'Mo 19', 'Tu 20'].map((day, idx) => {
                    const isToday = day === 'Tu 13'
                    return (
                      <div
                        key={idx}
                        style={{
                          padding: '8px 16px',
                          borderRadius: isToday ? '20px' : '8px',
                          background: isToday ? '#ff4d4f' : '#f5f5f5',
                          color: isToday ? '#fff' : '#595959',
                          fontSize: '13px',
                          fontWeight: isToday ? 600 : 400,
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                        }}
                        onMouseEnter={(e) => {
                          if (!isToday) {
                            e.currentTarget.style.background = '#e8e8e8'
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isToday) {
                            e.currentTarget.style.background = '#f5f5f5'
                          }
                        }}
                      >
                        {day}
                      </div>
                    )
                  })}
                  <div style={{
                    marginLeft: 'auto',
                    padding: '8px 12px',
                    background: '#f5f5f5',
                    borderRadius: '8px',
                    fontSize: '12px',
                    color: '#8c8c8c',
                  }}>
                    August 12-2021
                  </div>
                </div>
              </Card>
            </Col>

            {/* Stats Cards */}
            <Col xs={24} sm={12} md={6}>
              <StatsCard
                title="Tổng tương tác"
                value={totalInteractions}
                icon={<UserOutlined />}
                color="#1890ff"
                change={2.5}
                changeType="up"
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <StatsCard
                title="Khách hàng"
                value={uniqueCustomers}
                icon={<TeamOutlined />}
                color="#52c41a"
                change={1.2}
                changeType="up"
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <StatsCard
                title="Độ tuổi TB"
                value={avgAge.toFixed(1)}
                suffix=" tuổi"
                icon={<SmileOutlined />}
                color="#faad14"
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <StatsCard
                title="Quảng cáo"
                value={totalAds}
                icon={<VideoCameraOutlined />}
                color="#722ed1"
              />
            </Col>
          </Row>

        </Col>

        {/* Right Column - Profile & Actions */}
        <Col xs={24} lg={8}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <ProfileCard />
            <QuickActions />
            
            <WeeklyInvoice />
          </Space>
        </Col>
      </Row>

    </div>
  )
}

export default DashboardPage
