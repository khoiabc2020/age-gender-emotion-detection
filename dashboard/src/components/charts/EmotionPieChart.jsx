import React from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

const GRADIENT_COLORS = [
  { start: '#667eea', end: '#764ba2' },
  { start: '#f093fb', end: '#f5576c' },
  { start: '#4facfe', end: '#00f2fe' },
  { start: '#fa709a', end: '#fee140' },
  { start: '#30cfd0', end: '#330867' },
  { start: '#a8edea', end: '#fed6e3' },
  { start: '#ffecd2', end: '#fcb69f' },
]

const EmotionPieChart = ({ data = [] }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8 animate-pulse">
        <div className="text-sm">Chưa có dữ liệu</div>
      </div>
    )
  }

  const emotionLabels = {
    'happy': 'Vui vẻ',
    'sad': 'Buồn',
    'angry': 'Tức giận',
    'fear': 'Sợ hãi',
    'surprise': 'Ngạc nhiên',
    'neutral': 'Bình thường',
    'disgust': 'Khó chịu',
  }

  const chartData = data.map((item, index) => ({
    ...item,
    label: emotionLabels[item.emotion] || item.emotion,
    color: GRADIENT_COLORS[index % GRADIENT_COLORS.length],
  }))

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0]
      return (
        <div
          className="bg-white p-3 rounded-lg shadow-lg border border-gray-200"
          style={{ borderRadius: '12px' }}
        >
          <p className="font-semibold text-gray-800">{data.payload.label}</p>
          <p className="text-sm text-gray-600">
            Số lượng: <span className="font-semibold">{data.value}</span>
          </p>
          <p className="text-sm text-gray-600">
            Tỷ lệ: <span className="font-semibold">{((data.value / data.payload.total) * 100).toFixed(1)}%</span>
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <defs>
          {chartData.map((entry, index) => (
            <linearGradient key={`gradient-${index}`} id={`gradient-${index}`} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={entry.color.start} />
              <stop offset="100%" stopColor={entry.color.end} />
            </linearGradient>
          ))}
        </defs>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ label, percent }) => `${label} ${(percent * 100).toFixed(0)}%`}
          outerRadius={100}
          innerRadius={40}
          fill="#8884d8"
          dataKey="count"
          nameKey="label"
          animationDuration={1000}
          animationEasing="ease-out"
        >
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={`url(#gradient-${index})`}
              stroke="white"
              strokeWidth={2}
            />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ paddingTop: '20px' }}
          iconType="circle"
          formatter={(value) => <span style={{ color: '#64748b' }}>{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

export default EmotionPieChart
