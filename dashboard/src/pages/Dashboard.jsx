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

const { Title, Text } = Typography

const DashboardPage = () => {
  const dispatch = useAppDispatch()
  const { stats, ageByHour, emotionDistribution, loading } = useAppSelector(
    (state) => state.analytics
  )

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
    <div className="animate-fade-in" style={{ padding: '0' }}>
      {/* Top Bar with Search */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '24px',
        flexWrap: 'wrap',
        gap: '16px',
      }}>
        <Space size="large">
          <Title 
            level={2} 
            style={{ 
              margin: 0,
              color: 'var(--text-primary)',
              fontSize: '28px',
              fontWeight: 700,
            }}
          >
            Dashboard
          </Title>
        </Space>
        <Space>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            {new Date().toLocaleDateString('vi-VN', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </Text>
        </Space>
      </div>

      {/* Main Content Grid */}
      <Row gutter={[24, 24]}>
        {/* Left Column - Main Stats */}
        <Col xs={24} lg={16}>
          <Row gutter={[16, 16]}>
            {/* Greeting Card */}
            <Col xs={24} md={12}>
              <Card
                style={{
                  borderRadius: '12px',
                  border: 'none',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: '#fff',
                  height: '100%',
                }}
                bodyStyle={{ padding: '24px' }}
              >
                <Text style={{ color: 'rgba(255,255,255,0.9)', fontSize: '14px', display: 'block', marginBottom: '8px' }}>
                  Chào mừng trở lại
                </Text>
                <Title 
                  level={1} 
                  style={{ 
                    margin: 0, 
                    color: '#fff', 
                    fontSize: '36px',
                    fontWeight: 700,
                  }}
                >
                  {totalInteractions.toLocaleString()}
                </Title>
                <Space style={{ marginTop: '8px' }}>
                  <Text style={{ color: 'rgba(255,255,255,0.8)', fontSize: '12px' }}>
                    0.00% (0) 24h qua
                  </Text>
                </Space>
              </Card>
            </Col>

            {/* Spending Progress Card */}
            <Col xs={24} md={12}>
              <Card
                style={{
                  borderRadius: '12px',
                  border: 'none',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%)',
                  height: '100%',
                }}
                bodyStyle={{ padding: '24px', textAlign: 'center' }}
              >
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ 
                    width: '120px', 
                    height: '120px', 
                    margin: '0 auto',
                    position: 'relative',
                  }}>
                    <div style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      textAlign: 'center',
                    }}>
                      <Text style={{ 
                        color: '#fff', 
                        fontSize: '24px', 
                        fontWeight: 700,
                        display: 'block',
                      }}>
                        {spentPercentage.toFixed(0)}%
                      </Text>
                      <Text style={{ 
                        color: 'rgba(255,255,255,0.9)', 
                        fontSize: '12px',
                        display: 'block',
                      }}>
                        Đã sử dụng
                      </Text>
                    </div>
                  </div>
                </div>
                <div>
                  <Text style={{ color: 'rgba(255,255,255,0.9)', fontSize: '12px', display: 'block', marginBottom: '4px' }}>
                    Giới hạn hàng ngày {dailyLimit.toLocaleString()}
                  </Text>
                  <Text style={{ color: '#fff', fontSize: '16px', fontWeight: 600, display: 'block' }}>
                    Đã dùng hôm nay {usedToday.toLocaleString()}
                  </Text>
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

          <Divider style={{ margin: '24px 0' }} />

          {/* Charts Section */}
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card
                title="Phân bố Độ tuổi theo Giờ"
                className="card-hover"
                style={{
                  borderRadius: '12px',
                  border: 'none',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                }}
              >
                <AgeChart data={ageByHour} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card
                title="Phân bố Cảm xúc"
                className="card-hover"
                style={{
                  borderRadius: '12px',
                  border: 'none',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                }}
              >
                <EmotionPieChart data={emotionDistribution} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card
                title="Phân bố Giới tính"
                className="card-hover"
                style={{
                  borderRadius: '12px',
                  border: 'none',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                }}
              >
                <GenderChart data={stats?.gender_distribution} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card
                title="Hiệu suất Quảng cáo"
                className="card-hover"
                style={{
                  borderRadius: '12px',
                  border: 'none',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                }}
              >
                <AdPerformanceChart data={stats?.top_ads} />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Right Column - Profile & Actions */}
        <Col xs={24} lg={8}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <ProfileCard />
            <QuickActions />
            
            {/* Weekly Summary Card */}
            <Card
              title="Tóm tắt Tuần"
              style={{
                borderRadius: '12px',
                border: 'none',
                boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
              }}
            >
              <div style={{ marginBottom: '16px' }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Từ {new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toLocaleDateString('vi-VN')} - {new Date().toLocaleDateString('vi-VN')}
                </Text>
              </div>
              <div style={{ height: '200px', display: 'flex', alignItems: 'flex-end', gap: '8px' }}>
                {[65, 45, 80, 55, 70, 60, 75].map((value, index) => (
                  <div
                    key={index}
                    style={{
                      flex: 1,
                      height: `${value}%`,
                      background: index % 2 === 0 
                        ? 'linear-gradient(180deg, #52c41a 0%, #73d13d 100%)'
                        : 'linear-gradient(180deg, #ff4d4f 0%, #ff7875 100%)',
                      borderRadius: '4px 4px 0 0',
                      minHeight: '20px',
                    }}
                    title={`${value}%`}
                  />
                ))}
              </div>
            </Card>
          </Space>
        </Col>
      </Row>

      {/* Charts Section */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card
            title="Phân bố Độ tuổi theo Giờ"
            className="card-hover"
          >
            <AgeChart data={ageByHour} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title="Phân bố Cảm xúc"
            className="card-hover"
          >
            <EmotionPieChart data={emotionDistribution} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title="Phân bố Giới tính"
            className="card-hover"
          >
            <GenderChart data={stats?.gender_distribution} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title="Hiệu suất Quảng cáo"
            className="card-hover"
          >
            <AdPerformanceChart data={stats?.top_ads} />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default DashboardPage
