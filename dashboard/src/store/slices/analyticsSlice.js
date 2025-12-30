import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import api from '../../services/api'

export const fetchStats = createAsyncThunk(
  'analytics/fetchStats',
  async (hours = 24) => {
    const response = await api.get('/api/v1/analytics/stats', {
      params: { hours },
    })
    return response.data
  }
)

export const fetchAgeByHour = createAsyncThunk(
  'analytics/fetchAgeByHour',
  async (hours = 24) => {
    const response = await api.get('/api/v1/analytics/age-by-hour', {
      params: { hours },
    })
    return response.data
  }
)

export const fetchEmotionDistribution = createAsyncThunk(
  'analytics/fetchEmotionDistribution',
  async (hours = 24) => {
    const response = await api.get('/api/v1/analytics/emotion-distribution', {
      params: { hours },
    })
    return response.data
  }
)

export const fetchAdPerformance = createAsyncThunk(
  'analytics/fetchAdPerformance',
  async (hours = 24) => {
    const response = await api.get('/api/v1/analytics/ad-performance', {
      params: { hours },
    })
    return response.data
  }
)

const analyticsSlice = createSlice({
  name: 'analytics',
  initialState: {
    stats: null,
    ageByHour: [],
    emotionDistribution: [],
    adPerformance: [],
    loading: false,
    error: null,
  },
  reducers: {
    clearData: (state) => {
      state.stats = null
      state.ageByHour = []
      state.emotionDistribution = []
      state.adPerformance = []
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchStats.pending, (state) => {
        state.loading = true
      })
      .addCase(fetchStats.fulfilled, (state, action) => {
        state.loading = false
        state.stats = action.payload
      })
      .addCase(fetchStats.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message
      })
      .addCase(fetchAgeByHour.fulfilled, (state, action) => {
        state.ageByHour = action.payload
      })
      .addCase(fetchEmotionDistribution.fulfilled, (state, action) => {
        state.emotionDistribution = action.payload
      })
      .addCase(fetchAdPerformance.fulfilled, (state, action) => {
        state.adPerformance = action.payload
      })
  },
})

export const { clearData } = analyticsSlice.actions
export default analyticsSlice.reducer

