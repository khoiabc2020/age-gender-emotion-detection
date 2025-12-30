import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import api from '../../services/api'

export const fetchDevices = createAsyncThunk(
  'devices/fetchDevices',
  async () => {
    const response = await api.get('/api/v1/dashboard/devices')
    return response.data
  }
)

const devicesSlice = createSlice({
  name: 'devices',
  initialState: {
    devices: [],
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchDevices.pending, (state) => {
        state.loading = true
      })
      .addCase(fetchDevices.fulfilled, (state, action) => {
        state.loading = false
        state.devices = action.payload
      })
      .addCase(fetchDevices.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message
      })
  },
})

export default devicesSlice.reducer

