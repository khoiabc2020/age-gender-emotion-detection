import React from 'react'
import { Card, Progress, Typography, Space } from 'antd'
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons'

const { Text, Title } = Typography

const StatsCard = ({ 
  title, 
  value, 
  prefix, 
  suffix, 
  change, 
  changeType = 'up',
  icon,
  color = '#1890ff',
  progress,
  progressLabel 
}) => {
  const changeColor = changeType === 'up' ? '#52c41a' : '#ff4d4f'
  const ChangeIcon = changeType === 'up' ? ArrowUpOutlined : ArrowDownOutlined

  return (
    <Card
      className="stats-card-modern"
      style={{
        borderRadius: '12px',
        border: 'none',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        background: 'var(--bg-primary)',
        height: '100%',
      }}
      bodyStyle={{ padding: '20px' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
        <div>
          <Text type="secondary" style={{ fontSize: '13px', display: 'block', marginBottom: '8px' }}>
            {title}
          </Text>
          <Title 
            level={2} 
            style={{ 
              margin: 0, 
              fontSize: '32px', 
              fontWeight: 700,
              color: color,
            }}
          >
            {prefix}{value}{suffix}
          </Title>
        </div>
        {icon && (
          <div
            style={{
              width: '48px',
              height: '48px',
              borderRadius: '12px',
              background: `${color}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px',
              color: color,
            }}
          >
            {icon}
          </div>
        )}
      </div>

      {change !== undefined && (
        <Space style={{ marginTop: '8px' }}>
          <ChangeIcon style={{ color: changeColor, fontSize: '12px' }} />
          <Text style={{ color: changeColor, fontSize: '13px', fontWeight: 500 }}>
            {change}%
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            so với 24h trước
          </Text>
        </Space>
      )}

      {progress !== undefined && (
        <div style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <Text type="secondary" style={{ fontSize: '12px' }}>{progressLabel}</Text>
            <Text style={{ fontSize: '12px', fontWeight: 600, color: color }}>
              {progress}%
            </Text>
          </div>
          <Progress
            percent={progress}
            showInfo={false}
            strokeColor={color}
            trailColor="rgba(0,0,0,0.06)"
            style={{ margin: 0 }}
          />
        </div>
      )}
    </Card>
  )
}

export default StatsCard
