import React, { useEffect, useState } from 'react'
import { Card, Select, Row, Col, Table, Typography } from 'antd'
import { useAppDispatch, useAppSelector } from '../store/hooks'
import { useTheme } from '../components/layout/ThemeProvider'
import {
  fetchStats,
  fetchAgeByHour,
  fetchEmotionDistribution,
  fetchAdPerformance,
} from '../store/slices/analyticsSlice'
import AgeChart from '../components/charts/AgeChart'
import EmotionPieChart from '../components/charts/EmotionPieChart'
import AdPerformanceChart from '../components/charts/AdPerformanceChart'

const { Option } = Select
const { Title } = Typography

const AnalyticsPage = () => {
  const [timeRange, setTimeRange] = useState(24)
  const dispatch = useAppDispatch()
  const { darkMode } = useTheme()
  const {
    stats,
    ageByHour,
    emotionDistribution,
    adPerformance,
    loading,
  } = useAppSelector((state) => state.analytics)

  useEffect(() => {
    dispatch(fetchStats(timeRange))
    dispatch(fetchAgeByHour(timeRange))
    dispatch(fetchEmotionDistribution(timeRange))
    dispatch(fetchAdPerformance(timeRange))
  }, [dispatch, timeRange])

  const adPerformanceColumns = [
    {
      title: 'Mã Quảng cáo',
      dataIndex: 'ad_id',
      key: 'ad_id',
      render: (text) => <span style={{ 
        fontWeight: 600, 
        color: darkMode ? '#ffffff' : '#262626' 
      }}>{text}</span>,
    },
    {
      title: 'Số lần hiển thị',
      dataIndex: 'display_count',
      key: 'display_count',
      sorter: (a, b) => a.display_count - b.display_count,
      render: (value) => (
        <span className="font-semibold" style={{ color: '#667eea' }}>
          {value || 0}
        </span>
      ),
    },
    {
      title: 'Độ tuổi TB',
      dataIndex: 'avg_age',
      key: 'avg_age',
      render: (value) => (
        <span style={{ 
          color: darkMode ? 'rgba(255, 255, 255, 0.7)' : '#595959' 
        }}>
          {value ? `${value.toFixed(1)} tuổi` : 'N/A'}
        </span>
      ),
    },
  ]

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-6 flex justify-between items-center">
        <div>
          <Title 
            level={2} 
            className="page-title mb-2"
            style={{ 
              color: darkMode ? '#ffffff' : '#1a1a1a',
              fontWeight: 700,
              fontSize: '28px',
            }}
          >
            Phân tích Chi tiết
          </Title>
          <p className="page-description" style={{ 
            color: darkMode ? '#ffffff' : '#434343',
            fontSize: '15px',
            marginTop: '8px',
            opacity: darkMode ? 0.95 : 1,
          }}>
            Phân tích sâu về hành vi và hiệu suất quảng cáo
          </p>
        </div>
        <Select
          value={timeRange}
          onChange={setTimeRange}
          style={{ width: 200 }}
          size="large"
          className="rounded-lg"
        >
          <Option value={1}>1 giờ qua</Option>
          <Option value={6}>6 giờ qua</Option>
          <Option value={12}>12 giờ qua</Option>
          <Option value={24}>24 giờ qua</Option>
          <Option value={48}>48 giờ qua</Option>
          <Option value={72}>72 giờ qua</Option>
        </Select>
      </div>

      {/* Charts */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card
            title={
              <span className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                Phân bố Độ tuổi theo Giờ
              </span>
            }
            loading={loading}
            className="card-hover border-0 shadow-lg"
            style={{ 
              borderRadius: '20px', 
              background: darkMode ? '#252836' : 'rgba(255, 255, 255, 0.95)',
              border: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : 'none'
            }}
          >
            <AgeChart data={ageByHour} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title={
              <span className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                Phân bố Cảm xúc
              </span>
            }
            loading={loading}
            className="card-hover border-0 shadow-lg"
            style={{ 
              borderRadius: '20px', 
              background: darkMode ? '#252836' : 'rgba(255, 255, 255, 0.95)',
              border: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : 'none'
            }}
          >
            <EmotionPieChart data={emotionDistribution} />
          </Card>
        </Col>
        <Col xs={24}>
          <Card
            title={
              <span className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                Hiệu suất Quảng cáo
              </span>
            }
            loading={loading}
            className="card-hover border-0 shadow-lg"
            style={{ 
              borderRadius: '20px', 
              background: darkMode ? '#252836' : 'rgba(255, 255, 255, 0.95)',
              border: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : 'none'
            }}
          >
            <AdPerformanceChart data={adPerformance} />
          </Card>
        </Col>
        <Col xs={24}>
          <Card
            title={
              <span className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                Chi tiết Quảng cáo
              </span>
            }
            loading={loading}
            className="card-hover border-0 shadow-lg"
            style={{ 
              borderRadius: '20px', 
              background: darkMode ? '#252836' : 'rgba(255, 255, 255, 0.95)',
              border: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : 'none'
            }}
          >
            <Table
              dataSource={adPerformance}
              columns={adPerformanceColumns}
              rowKey="ad_id"
              pagination={{ pageSize: 10 }}
              className="custom-table"
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default AnalyticsPage
