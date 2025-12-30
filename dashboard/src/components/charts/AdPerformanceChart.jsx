import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const AdPerformanceChart = ({ data = [] }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8 animate-pulse">
        <div className="text-sm">Chưa có dữ liệu</div>
      </div>
    )
  }

  // Limit to top 10 and format data
  const chartData = data
    .slice(0, 10)
    .map((item, index) => ({
      ...item,
      name: item.ad_id?.replace(/_/g, ' ') || `Quảng cáo ${index + 1}`,
      displayCount: item.count || item.display_count || 0,
    }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
      >
        <defs>
          <linearGradient id="colorBar" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#667eea" stopOpacity={0.9} />
            <stop offset="95%" stopColor="#764ba2" stopOpacity={0.9} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.5} />
        <XAxis
          dataKey="name"
          angle={-45}
          textAnchor="end"
          height={100}
          stroke="#94a3b8"
          style={{ fontSize: '11px' }}
        />
        <YAxis
          stroke="#94a3b8"
          style={{ fontSize: '12px' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            border: 'none',
            borderRadius: '12px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          }}
          labelStyle={{ color: '#1e293b', fontWeight: 600 }}
        />
        <Legend
          wrapperStyle={{ paddingTop: '20px' }}
          iconType="square"
        />
        <Bar
          dataKey="displayCount"
          fill="url(#colorBar)"
          name="Số lần hiển thị"
          radius={[8, 8, 0, 0]}
          animationDuration={1000}
          animationEasing="ease-out"
        />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default AdPerformanceChart
