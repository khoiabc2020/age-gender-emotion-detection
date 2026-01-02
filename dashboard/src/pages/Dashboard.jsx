import React, { useEffect } from 'react'
import { Row, Col, Card, Statistic, Spin, Typography } from 'antd'
import {
  UserOutlined,
  TeamOutlined,
  SmileOutlined,
  VideoCameraOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
} from '@ant-design/icons'
import { useAppDispatch, useAppSelector } from '../store/hooks'
import { fetchStats, fetchAgeByHour, fetchEmotionDistribution } from '../store/slices/analyticsSlice'
import AgeChart from '../components/charts/AgeChart'
import EmotionPieChart from '../components/charts/EmotionPieChart'
import GenderChart from '../components/charts/GenderChart'
import AdPerformanceChart from '../components/charts/AdPerformanceChart'

const { Title } = Typography

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

  const statCards = [
    {
      title: 'Tổng Tương tác',
      value: stats?.total_interactions || 0,
      icon: <UserOutlined />,
      gradient: 'bg-gradient-primary',
      prefix: null,
      suffix: null,
    },
    {
      title: 'Khách hàng',
      value: stats?.unique_customers || 0,
      icon: <TeamOutlined />,
      gradient: 'bg-gradient-success',
      prefix: null,
      suffix: null,
    },
    {
      title: 'Độ tuổi TB',
      value: stats?.avg_age || 0,
      icon: <SmileOutlined />,
      gradient: 'bg-gradient-warning',
      prefix: null,
      suffix: 'tuổi',
      precision: 1,
    },
    {
      title: 'Quảng cáo',
      value: stats?.top_ads?.length || 0,
      icon: <VideoCameraOutlined />,
      gradient: 'bg-gradient-secondary',
      prefix: null,
      suffix: null,
    },
  ]

  const iconColors = [
    { bg: '#e6f7ff', color: '#1890ff' },
    { bg: '#f6ffed', color: '#52c41a' },
    { bg: '#fffbe6', color: '#faad14' },
    { bg: '#f0f5ff', color: '#2f54eb' },
  ]

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="page-header">
        <Title level={2} className="page-title">
          Tổng quan Hệ thống
        </Title>
        <p className="page-description">Theo dõi và phân tích dữ liệu khách hàng realtime</p>
      </div>

      {/* Key Metrics Cards */}
      <Row gutter={[16, 16]} className="mb-6">
        {statCards.map((stat, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <Card className="card-hover stats-card">
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
                <div
                  className="stats-card-icon"
                  style={{
                    background: iconColors[index].bg,
                    color: iconColors[index].color,
                    flexShrink: 0,
                  }}
                >
                  {stat.icon}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <Statistic
                    title={<span style={{ color: '#8c8c8c', fontSize: '14px' }}>{stat.title}</span>}
                    value={stat.value}
                    prefix={stat.prefix}
                    suffix={stat.suffix}
                    precision={stat.precision}
                    valueStyle={{
                      fontSize: '24px',
                      fontWeight: 600,
                      color: '#262626',
                    }}
                  />
                </div>
              </div>
            </Card>
          </Col>
        ))}
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
