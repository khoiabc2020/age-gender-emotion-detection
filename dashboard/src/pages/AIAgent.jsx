import React, { useState, useEffect, useRef } from 'react'
import { Card, Input, Button, Typography, Space, Spin, message, Select, Divider, Tag, List, Avatar, Row, Col } from 'antd'
import {
  RobotOutlined,
  SendOutlined,
  FileTextOutlined,
  BulbOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import api from '../services/api'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { Option } = Select

const AIAgentPage = () => {
  const [question, setQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [reportGenerating, setReportGenerating] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [report, setReport] = useState(null)
  const [timeRange, setTimeRange] = useState(24)
  const [aiStatus, setAiStatus] = useState(null)
  const chatEndRef = useRef(null)

  useEffect(() => {
    checkAIStatus()
  }, [])

  const checkAIStatus = async () => {
    try {
      const response = await api.get('/api/v1/ai/status')
      setAiStatus(response.data)
    } catch (error) {
      console.error('Failed to check AI status:', error)
    }
  }

  const handleChat = async () => {
    if (!question.trim()) {
      message.warning('Vui lòng nhập câu hỏi!')
      return
    }

    setLoading(true)
    const userQuestion = question
    setQuestion('')

    // Add user message to history
    const userMessage = {
      type: 'user',
      content: userQuestion,
      timestamp: new Date().toLocaleTimeString(),
    }
    setChatHistory((prev) => [...prev, userMessage])

    try {
      const response = await api.post('/api/v1/ai/chat', {
        question: userQuestion,
        time_range_hours: timeRange,
      })

      const aiMessage = {
        type: 'ai',
        content: response.data.answer,
        timestamp: new Date().toLocaleTimeString(),
      }
      setChatHistory((prev) => [...prev, aiMessage])
      // Auto scroll to bottom
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (error) {
      message.error('Lỗi khi gửi câu hỏi. Vui lòng kiểm tra cấu hình AI.')
      setChatHistory((prev) => prev.slice(0, -1)) // Remove user message on error
    } finally {
      setLoading(false)
    }
  }

  const handleAnalyze = async () => {
    setAnalyzing(true)
    try {
      const response = await api.post(`/api/v1/ai/analyze?time_range_hours=${timeRange}`)
      setAnalysis(response.data)
      message.success('Phân tích hoàn tất!')
    } catch (error) {
      message.error('Lỗi khi phân tích. Vui lòng kiểm tra cấu hình AI.')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleGenerateReport = async () => {
    setReportGenerating(true)
    try {
      const response = await api.post('/api/v1/ai/generate-report', {
        time_range_hours: timeRange,
        include_charts: true,
      })
      setReport(response.data.report)
      message.success('Báo cáo đã được tạo!')
    } catch (error) {
      message.error('Lỗi khi tạo báo cáo. Vui lòng kiểm tra cấu hình AI.')
    } finally {
      setReportGenerating(false)
    }
  }

  const quickQuestions = [
    'Phân tích xu hướng khách hàng trong 24h qua?',
    'Nhóm tuổi nào chiếm đa số?',
    'Cảm xúc phổ biến nhất là gì?',
    'Quảng cáo nào hiệu quả nhất?',
    'Đề xuất cải thiện hệ thống?',
  ]

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <Title 
              level={2} 
              className="page-title mb-2" 
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '12px',
                color: 'var(--text-primary)'
              }}
            >
              <RobotOutlined style={{ fontSize: '28px', color: '#1890ff' }} />
              AI Agent - Trợ lý Phân tích
            </Title>
            <p 
              className="page-description"
              style={{ color: 'var(--text-secondary)' }}
            >
              Chat với dữ liệu và nhận insights tự động từ AI
            </p>
          </div>
          {aiStatus && (
            <Tag
              color={aiStatus.available ? 'success' : 'warning'}
              icon={aiStatus.available ? <CheckCircleOutlined /> : <LoadingOutlined />}
              className="text-base px-4 py-2"
            >
              {aiStatus.available
                ? `AI ${aiStatus.provider === 'both' ? 'Ready (Both)' : aiStatus.provider === 'google_ai' ? 'Ready (Google)' : 'Ready (ChatGPT)'}`
                : 'Chưa cấu hình'}
            </Tag>
          )}
        </div>
      </div>

      {/* AI Status Info */}
      {aiStatus && !aiStatus.available && (
        <Card
          className="mb-6"
          style={{ 
            borderColor: '#fff7e6',
            backgroundColor: '#fffbe6',
          }}
        >
          <div className="flex items-center gap-3">
            <ThunderboltOutlined className="text-2xl text-orange-500" />
            <div>
              <Text strong className="text-orange-800">
                AI Agent chưa được cấu hình
              </Text>
              <Paragraph className="text-orange-600 mb-0 mt-1">
                Vui lòng thêm API keys trong Settings để sử dụng tính năng AI.
                Hỗ trợ Google AI (Gemini) và ChatGPT.
              </Paragraph>
            </div>
          </div>
        </Card>
      )}

      <Row gutter={[16, 16]}>
        {/* Chat Section */}
        <Col xs={24} lg={14}>
          <Card
            className="card-hover h-full"
            title={
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                <RobotOutlined /> Chat với AI
              </span>
            }
            extra={
              <Select
                value={timeRange}
                onChange={setTimeRange}
                style={{ width: 120 }}
                size="small"
              >
                <Option value={1}>1 giờ</Option>
                <Option value={6}>6 giờ</Option>
                <Option value={24}>24 giờ</Option>
                <Option value={48}>48 giờ</Option>
                <Option value={72}>72 giờ</Option>
              </Select>
            }
          >
            {/* Chat History */}
            <div
              className="mb-4 p-4 rounded-lg chat-container"
              style={{
                height: '400px',
                overflowY: 'auto',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
              }}
            >
              {chatHistory.length === 0 ? (
                <div className="text-center py-8">
                  <RobotOutlined className="text-4xl mb-3" style={{ color: 'var(--text-secondary)' }} />
                  <p style={{ color: 'var(--text-primary)', fontSize: '15px', fontWeight: 500 }}>Bắt đầu trò chuyện với AI Agent</p>
                  <p className="text-sm mt-2" style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>Hoặc chọn câu hỏi nhanh bên dưới</p>
                </div>
              ) : (
                <Space direction="vertical" size="large" className="w-full">
                  {chatHistory.map((msg, index) => (
                    <div
                      key={index}
                      className={`flex gap-3 ${
                        msg.type === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      {msg.type === 'ai' && (
                        <Avatar
                          icon={<RobotOutlined />}
                          style={{
                            background: '#1890ff',
                          }}
                        />
                      )}
                      <div
                        className={`max-w-[80%] p-3 rounded-lg ${
                          msg.type === 'user'
                            ? 'bg-blue-600 text-white'
                            : 'bg-white border'
                        }`}
                        style={{
                          borderRadius: '8px',
                          borderColor: 'var(--border-color)',
                        }}
                      >
                        <Paragraph 
                          className="mb-1" 
                          style={{ 
                            color: msg.type === 'user' ? 'white' : 'var(--text-primary)',
                            fontSize: '14px',
                            lineHeight: '1.6'
                          }}
                        >
                          {msg.content}
                        </Paragraph>
                        <Text
                          className="text-xs"
                          style={{ 
                            color: msg.type === 'user' ? 'rgba(255,255,255,0.9)' : 'var(--text-secondary)',
                            fontSize: '12px'
                          }}
                        >
                          {msg.timestamp}
                        </Text>
                      </div>
                      {msg.type === 'user' && (
                        <Avatar
                          style={{
                            background: '#52c41a',
                          }}
                        >
                          U
                        </Avatar>
                      )}
                    </div>
                  ))}
                  {loading && (
                    <div className="flex justify-start gap-3">
                      <Avatar
                        icon={<RobotOutlined />}
                        style={{
                          background: '#1890ff',
                        }}
                      />
                      <div className="bg-white border p-3 rounded-lg" style={{ borderColor: 'var(--border-color)' }}>
                        <Spin size="small" />
                        <span className="ml-2" style={{ color: 'var(--text-primary)', fontSize: '14px' }}>AI đang suy nghĩ...</span>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </Space>
              )}
            </div>

            {/* Quick Questions */}
            <div className="mb-4">
              <Text className="text-sm mb-2 block" style={{ color: 'var(--text-primary)', fontWeight: 500, fontSize: '14px' }}>Câu hỏi nhanh:</Text>
              <Space wrap>
                {quickQuestions.map((q, index) => (
                  <Button
                    key={index}
                    size="small"
                    type="dashed"
                    onClick={() => setQuestion(q)}
                    className="rounded-lg quick-question-btn"
                    style={{
                      borderColor: 'rgba(255, 255, 255, 0.2)',
                      color: 'rgba(255, 255, 255, 0.9)',
                      background: 'rgba(255, 255, 255, 0.05)',
                      fontSize: '13px',
                      height: '32px',
                      padding: '0 16px',
                    }}
                  >
                    {q}
                  </Button>
                ))}
              </Space>
            </div>

            {/* Input */}
            <Space.Compact className="w-full">
              <TextArea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Nhập câu hỏi của bạn..."
                rows={2}
                onPressEnter={(e) => {
                  if (!e.shiftKey) {
                    e.preventDefault()
                    handleChat()
                  }
                }}
                className="rounded-l-lg"
              />
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleChat}
                loading={loading}
                className="rounded-r-lg"
                style={{
                  height: 'auto',
                }}
              >
                Gửi
              </Button>
            </Space.Compact>
          </Card>
        </Col>

        {/* Analysis & Report Section */}
        <Col xs={24} lg={10}>
          <Space direction="vertical" size="large" className="w-full">
            {/* Quick Analyze */}
            <Card
              className="card-hover"
              title={
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                  <BulbOutlined /> Phân tích Nhanh
                </span>
              }
            >
              <Button
                type="primary"
                block
                icon={<BulbOutlined />}
                onClick={handleAnalyze}
                loading={analyzing}
                className="rounded-lg mb-3"
                style={{
                  height: '48px',
                  fontSize: '16px',
                  fontWeight: 600,
                }}
              >
                Phân tích Dữ liệu
              </Button>
              {analysis && (
                <div className="mt-4 space-y-3">
                  {analysis.insights && analysis.insights.length > 0 && (
                    <div>
                      <Text strong style={{ color: 'var(--text-primary)' }}>Insights:</Text>
                      <List
                        size="small"
                        dataSource={analysis.insights}
                        renderItem={(item) => (
                          <List.Item>
                            <Text className="text-sm">{item}</Text>
                          </List.Item>
                        )}
                      />
                    </div>
                  )}
                  {analysis.recommendations && analysis.recommendations.length > 0 && (
                    <div>
                      <Text strong style={{ color: 'var(--text-primary)' }}>Đề xuất:</Text>
                      <List
                        size="small"
                        dataSource={analysis.recommendations}
                        renderItem={(item) => (
                          <List.Item>
                            <Text className="text-sm">{item}</Text>
                          </List.Item>
                        )}
                      />
                    </div>
                  )}
                </div>
              )}
            </Card>

            {/* Generate Report */}
            <Card
              className="card-hover"
              title={
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                  <FileTextOutlined /> Báo cáo Tự động
                </span>
              }
            >
              <Button
                type="primary"
                block
                icon={<FileTextOutlined />}
                onClick={handleGenerateReport}
                loading={reportGenerating}
                className="rounded-lg"
                style={{
                  height: '48px',
                  fontSize: '16px',
                  fontWeight: 600,
                }}
              >
                Tạo Báo cáo
              </Button>
              {report && (
                <div
                  className="mt-4 p-4 rounded-lg"
                  style={{
                    maxHeight: '300px',
                    overflowY: 'auto',
                    whiteSpace: 'pre-wrap',
                    fontSize: '13px',
                    background: 'var(--bg-tertiary)',
                  }}
                >
                  <Paragraph style={{ color: 'var(--text-primary)' }}>{report}</Paragraph>
                </div>
              )}
            </Card>
          </Space>
        </Col>
      </Row>
    </div>
  )
}

export default AIAgentPage

