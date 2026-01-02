import React, { useEffect } from 'react'
import { Card, Table, Button, Space, Tag, Modal, Form, Input, InputNumber, Select, message, Typography, Row, Col } from 'antd'
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'
import api from '../services/api'
import { useTheme } from '../components/layout/ThemeProvider'

const { Option } = Select
const { Title } = Typography

const AdsManagementPage = () => {
  const [ads, setAds] = React.useState([])
  const [loading, setLoading] = React.useState(false)
  const [modalVisible, setModalVisible] = React.useState(false)
  const [editingAd, setEditingAd] = React.useState(null)
  const [form] = Form.useForm()
  const { darkMode } = useTheme()

  useEffect(() => {
    fetchAds()
  }, [])

  const fetchAds = async () => {
    setLoading(true)
    try {
      const response = await api.get('/api/v1/ads/')
      setAds(response.data)
    } catch (error) {
      message.error('Lỗi khi tải danh sách quảng cáo')
    } finally {
      setLoading(false)
    }
  }

  const handleAdd = () => {
    setEditingAd(null)
    form.resetFields()
    setModalVisible(true)
  }

  const handleEdit = (ad) => {
    setEditingAd(ad)
    form.setFieldsValue(ad)
    setModalVisible(true)
  }

  const handleDelete = async (adId) => {
    Modal.confirm({
      title: 'Xác nhận xóa',
      content: 'Bạn có chắc chắn muốn xóa quảng cáo này?',
      okText: 'Xóa',
      cancelText: 'Hủy',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          await api.delete(`/api/v1/ads/${adId}`)
          message.success('Xóa thành công')
          fetchAds()
        } catch (error) {
          message.error('Lỗi khi xóa quảng cáo')
        }
      },
    })
  }

  const handleSubmit = async (values) => {
    try {
      if (editingAd) {
        await api.put(`/api/v1/ads/${editingAd.id}`, values)
        message.success('Cập nhật thành công')
      } else {
        await api.post('/api/v1/ads/', values)
        message.success('Thêm thành công')
      }
      setModalVisible(false)
      fetchAds()
    } catch (error) {
      message.error('Lỗi khi lưu quảng cáo')
    }
  }

  const columns = [
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
      title: 'Tên',
      dataIndex: 'name',
      key: 'name',
      render: (text) => <span style={{ 
        color: darkMode ? 'rgba(255, 255, 255, 0.85)' : '#262626' 
      }}>{text}</span>,
    },
    {
      title: 'Độ tuổi',
      key: 'age_range',
      render: (_, record) => (
        <Tag color="blue">
          {record.target_age_min || '0'} - {record.target_age_max || '100'} tuổi
        </Tag>
      ),
    },
    {
      title: 'Giới tính',
      dataIndex: 'target_gender',
      key: 'target_gender',
      render: (gender) => (
        <Tag color={gender === 'male' ? 'blue' : gender === 'female' ? 'pink' : 'default'}>
          {gender === 'male' ? 'Nam' : gender === 'female' ? 'Nữ' : 'Tất cả'}
        </Tag>
      ),
    },
    {
      title: 'Độ ưu tiên',
      dataIndex: 'priority',
      key: 'priority',
      sorter: (a, b) => a.priority - b.priority,
      render: (value) => (
        <Tag color={value >= 8 ? 'red' : value >= 5 ? 'orange' : 'default'}>
          {value}/10
        </Tag>
      ),
    },
    {
      title: 'Thao tác',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button
            type="link"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
            className="text-blue-500"
          >
            Sửa
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record.ad_id)}
          >
            Xóa
          </Button>
        </Space>
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
            style={{ color: 'var(--text-primary)' }}
          >
            Quản lý Quảng cáo
          </Title>
          <p className="page-description" style={{ color: 'var(--text-secondary)' }}>Quản lý và cấu hình các quảng cáo hiển thị</p>
        </div>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={handleAdd}
          size="large"
          className="rounded-lg bg-gradient-primary border-0 shadow-lg hover:shadow-xl transition-all duration-300"
          style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            height: '48px',
            paddingLeft: '24px',
            paddingRight: '24px',
            fontSize: '16px',
            fontWeight: 600,
          }}
        >
          Thêm Quảng cáo
        </Button>
      </div>

      {/* Table */}
      <Card
        className="card-hover border-0 shadow-lg"
        style={{ 
          borderRadius: '20px', 
          background: darkMode ? '#252836' : 'rgba(255, 255, 255, 0.95)',
          border: darkMode ? '1px solid rgba(255, 255, 255, 0.08)' : 'none'
        }}
      >
        <Table
          dataSource={ads}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          className="custom-table"
        />
      </Card>

      {/* Modal */}
      <Modal
        title={
          <span className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
            {editingAd ? 'Sửa Quảng cáo' : 'Thêm Quảng cáo'}
          </span>
        }
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        onOk={() => form.submit()}
        width={700}
        okText="Lưu"
        cancelText="Hủy"
        okButtonProps={{
          className: 'rounded-lg bg-gradient-primary border-0',
          style: {
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          },
        }}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
        >
          <Form.Item
            name="ad_id"
            label="Mã Quảng cáo"
            rules={[{ required: true, message: 'Vui lòng nhập mã quảng cáo' }]}
          >
            <Input className="rounded-lg" placeholder="vd: coffee_morning" />
          </Form.Item>
          <Form.Item
            name="name"
            label="Tên"
            rules={[{ required: true, message: 'Vui lòng nhập tên' }]}
          >
            <Input className="rounded-lg" placeholder="Tên quảng cáo" />
          </Form.Item>
          <Form.Item name="description" label="Mô tả">
            <Input.TextArea rows={3} className="rounded-lg" placeholder="Mô tả quảng cáo" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="target_age_min" label="Độ tuổi tối thiểu">
                <InputNumber min={0} max={100} style={{ width: '100%' }} className="rounded-lg" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="target_age_max" label="Độ tuổi tối đa">
                <InputNumber min={0} max={100} style={{ width: '100%' }} className="rounded-lg" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="target_gender" label="Giới tính mục tiêu">
            <Select className="rounded-lg">
              <Option value="all">Tất cả</Option>
              <Option value="male">Nam</Option>
              <Option value="female">Nữ</Option>
            </Select>
          </Form.Item>
          <Form.Item name="priority" label="Độ ưu tiên (1-10)">
            <InputNumber min={1} max={10} style={{ width: '100%' }} className="rounded-lg" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AdsManagementPage
