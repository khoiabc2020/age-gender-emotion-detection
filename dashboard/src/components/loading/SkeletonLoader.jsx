import React from 'react'

const SkeletonLoader = ({ rows = 3, className = '' }) => {
  return (
    <div className={`space-y-4 ${className}`}>
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="skeleton h-4 rounded" style={{ width: `${100 - index * 10}%` }} />
      ))}
    </div>
  )
}

export default SkeletonLoader

