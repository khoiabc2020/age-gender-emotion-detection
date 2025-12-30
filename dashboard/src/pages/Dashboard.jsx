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
    dispatch(fetchStats(24))
    dispatch(fetchAgeByHour(24))
    dispatch(fetchEmotionDistribution(24))
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      dispatch(fetchStats(24))
      dispatch(fetchAgeByHour(24))
      dispatch(fetchEmotionDistribution(24))
    }, 30000)
    
    return () => clearInterval(interval)
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

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-6">
        <Title level={2} className="gradient-text mb-2">
          Tổng quan Hệ thống
        </Title>
        <p className="text-gray-500">Theo dõi và phân tích dữ liệu khách hàng realtime</p>
      </div>

      {/* Key Metrics Cards */}
      <Row gutter={[16, 16]} className="mb-6">
        {statCards.map((stat, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <Card
              className="card-hover enhanced-card border-0 shadow-lg animate-scale-in"
              style={{
                background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%)',
                borderRadius: '20px',
                animationDelay: `${index * 0.1}s`,
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <Statistic
                    title={<span className="text-gray-600 font-medium">{stat.title}</span>}
                    value={stat.value}
                    prefix={stat.prefix}
                    suffix={stat.suffix}
                    precision={stat.precision}
                    valueStyle={{
                      fontSize: '28px',
                      fontWeight: 700,
                      background: `linear-gradient(135deg, ${index === 0 ? '#667eea' : index === 1 ? '#4facfe' : index === 2 ? '#fa709a' : '#f093fb'} 0%, ${index === 0 ? '#764ba2' : index === 1 ? '#00f2fe' : index === 2 ? '#fee140' : '#f5576c'} 100%)`,
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                    }}
                  />
                </div>
                <div
                  className={`w-16 h-16 rounded-2xl flex items-center justify-center text-white text-2xl shadow-lg animate-float ${stat.gradient}`}
                  style={{
                    background: index === 0
                      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                      : index === 1
                      ? 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
                      : index === 2
                      ? 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
                      : 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                    animationDelay: `${index * 0.2}s`,
                    transition: 'transform 0.3s ease',
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1) rotate(5deg)'}
                  onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1) rotate(0deg)'}
                >
                  {stat.icon}
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
            title={
              <span className="text-lg font-semibold gradient-text">
                Phân bố Độ tuổi theo Giờ
              </span>
            }
            className="card-hover border-0 shadow-lg"
            style={{
              borderRadius: '20px',
              background: 'rgba(255, 255, 255, 0.95)',
            }}
          >
            <AgeChart data={ageByHour} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title={
              <span className="text-lg font-semibold gradient-text">
                Phân bố Cảm xúc
              </span>
            }
            className="card-hover border-0 shadow-lg"
            style={{
              borderRadius: '20px',
              background: 'rgba(255, 255, 255, 0.95)',
            }}
          >
            <EmotionPieChart data={emotionDistribution} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title={
              <span className="text-lg font-semibold gradient-text">
                Phân bố Giới tính
              </span>
            }
            className="card-hover border-0 shadow-lg"
            style={{
              borderRadius: '20px',
              background: 'rgba(255, 255, 255, 0.95)',
            }}
          >
            <GenderChart data={stats?.gender_distribution} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title={
              <span className="text-lg font-semibold gradient-text">
                Hiệu suất Quảng cáo
              </span>
            }
            className="card-hover border-0 shadow-lg"
            style={{
              borderRadius: '20px',
              background: 'rgba(255, 255, 255, 0.95)',
            }}
          >
            <AdPerformanceChart data={stats?.top_ads} />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default DashboardPage
