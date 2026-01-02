import React from 'react'
import { Card, Typography } from 'antd'
import { useTheme } from '../layout/ThemeProvider'

const { Text } = Typography

const WeeklyInvoice = () => {
  const { darkMode } = useTheme()
  const data = [
    { value: 65, date: '12 Oct' },
    { value: 45, date: '14 Oct' },
    { value: 80, date: '16 Oct' },
    { value: 55, date: '18 Oct' },
    { value: 70, date: '20 Oct' },
    { value: 60, date: '22 Oct' },
    { value: 75, date: '24 Nov' },
  ]

  const maxValue = Math.max(...data.map(d => d.value))

  return (
    <Card
      title={
        <span style={{ color: darkMode ? '#ffffff' : '#262626', fontWeight: 600 }}>
          Weekly Invoice
        </span>
      }
      style={{
        borderRadius: '16px',
        border: 'none',
        boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.3)' : '0 4px 12px rgba(0,0,0,0.1)',
        background: darkMode ? '#252836' : '#ffffff',
        border: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : 'none',
      }}
      headStyle={{
        borderBottom: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : '1px solid #f0f0f0',
        padding: '20px 24px',
        background: darkMode ? '#252836' : '#ffffff',
      }}
      bodyStyle={{ padding: '24px' }}
    >
      <div style={{ marginBottom: '16px' }}>
        <Text style={{ 
          fontSize: '12px',
          color: darkMode ? 'rgba(255, 255, 255, 0.7)' : '#8c8c8c'
        }}>
          From 12 Oct - 24 Nov
        </Text>
      </div>
      <div style={{ 
        height: '200px', 
        display: 'flex', 
        alignItems: 'flex-end', 
        gap: '12px',
        paddingTop: '20px',
      }}>
        {data.map((item, index) => (
          <div key={index} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
            <div
              style={{
                width: '100%',
                height: `${(item.value / maxValue) * 100}%`,
                minHeight: '20px',
                background: index % 2 === 0 
                  ? 'linear-gradient(180deg, #52c41a 0%, #73d13d 100%)'
                  : 'linear-gradient(180deg, #ff4d4f 0%, #ff7875 100%)',
                borderRadius: '6px 6px 0 0',
                transition: 'all 0.3s',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'scaleY(1.05)'
                e.currentTarget.style.opacity = '0.9'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scaleY(1)'
                e.currentTarget.style.opacity = '1'
              }}
              title={`${item.value}%`}
            />
            <Text style={{ 
              fontSize: '11px', 
              color: darkMode ? 'rgba(255, 255, 255, 0.7)' : '#8c8c8c' 
            }}>
              {item.date}
            </Text>
          </div>
        ))}
      </div>
    </Card>
  )
}

export default WeeklyInvoice
