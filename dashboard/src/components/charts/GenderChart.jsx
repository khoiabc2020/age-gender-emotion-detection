import React from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

const GenderChart = ({ data = {} }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="text-center text-gray-400 py-8 animate-pulse">
        <div className="text-sm">Chưa có dữ liệu</div>
      </div>
    )
  }

  const chartData = Object.entries(data).map(([name, value]) => ({
    name: name === 'male' ? 'Nam' : 'Nữ',
    value,
    color: name === 'male' 
      ? { start: '#667eea', end: '#764ba2' }
      : { start: '#f093fb', end: '#f5576c' },
  }))

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0]
      return (
        <div
          className="bg-white p-3 rounded-lg shadow-lg border border-gray-200"
          style={{ borderRadius: '12px' }}
        >
          <p className="font-semibold text-gray-800">{data.name}</p>
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

  const total = chartData.reduce((sum, item) => sum + item.value, 0)
  chartData.forEach(item => item.total = total)

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <defs>
          {chartData.map((entry, index) => (
            <linearGradient key={`gradient-${index}`} id={`gender-gradient-${index}`} x1="0" y1="0" x2="1" y2="1">
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
          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          outerRadius={100}
          innerRadius={50}
          fill="#8884d8"
          dataKey="value"
          animationDuration={1000}
          animationEasing="ease-out"
        >
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={`url(#gender-gradient-${index})`}
              stroke="white"
              strokeWidth={3}
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

export default GenderChart
