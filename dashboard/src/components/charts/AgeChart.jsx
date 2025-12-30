import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'

const AgeChart = ({ data = [] }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8 animate-pulse">
        <div className="text-sm">Chưa có dữ liệu</div>
      </div>
    )
  }

  // Sort data by hour
  const sortedData = [...data].sort((a, b) => a.hour - b.hour)

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart
          data={sortedData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
        <defs>
          <linearGradient id="colorAge" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#667eea" stopOpacity={0.8} />
            <stop offset="95%" stopColor="#764ba2" stopOpacity={0.1} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.5} />
        <XAxis
          dataKey="hour"
          label={{ value: 'Giờ', position: 'insideBottom', offset: -5, style: { fill: '#64748b' } }}
          stroke="#94a3b8"
          style={{ fontSize: '12px' }}
        />
        <YAxis
          label={{ value: 'Độ tuổi TB', angle: -90, position: 'insideLeft', style: { fill: '#64748b' } }}
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
          iconType="circle"
        />
        <Area
          type="monotone"
          dataKey="avg_age"
          stroke="url(#colorAge)"
          strokeWidth={3}
          fill="url(#colorAge)"
          name="Độ tuổi trung bình"
          animationDuration={1000}
          animationEasing="ease-out"
        />
      </AreaChart>
    </ResponsiveContainer>
    </div>
  )
}

export default AgeChart
