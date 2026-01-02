import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import api from '../../services/api'

export const login = createAsyncThunk(
  'auth/login',
  async ({ username, password }, { rejectWithValue }) => {
    try {
      const response = await api.post('/auth/login', { username, password })
      const { token } = response.data
      localStorage.setItem('token', token)
      return { token }
    } catch (error) {
      return rejectWithValue(error.response?.data?.message || 'Login failed')
    }
  }
)

const getStoredUser = () => {
  try {
    const userStr = localStorage.getItem('user')
    return userStr ? JSON.parse(userStr) : null
  } catch {
    return null
  }
}

// Safe localStorage access
const getInitialAuthState = () => {
  try {
    const token = localStorage.getItem('token')
    return {
      isAuthenticated: !!token,
      token: token,
      user: getStoredUser(),
      loading: false,
      error: null,
    }
  } catch (error) {
    console.warn('Failed to read from localStorage:', error)
    return {
      isAuthenticated: false,
      token: null,
      user: null,
      loading: false,
      error: null,
    }
  }
}

const authSlice = createSlice({
  name: 'auth',
  initialState: getInitialAuthState(),
  reducers: {
    login: (state, action) => {
      state.isAuthenticated = true
      state.token = action.payload.token
      state.user = action.payload.user
      localStorage.setItem('token', action.payload.token)
      localStorage.setItem('user', JSON.stringify(action.payload.user))
    },
    logout: (state) => {
      state.isAuthenticated = false
      state.token = null
      state.user = null
      localStorage.removeItem('token')
      localStorage.removeItem('user')
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(login.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(login.fulfilled, (state, action) => {
        state.loading = false
        state.isAuthenticated = true
        state.token = action.payload.token
        state.user = action.payload.user
      })
      .addCase(login.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload
      })
  },
})

export const { logout, clearError } = authSlice.actions
export default authSlice.reducer

