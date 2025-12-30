import { configureStore } from '@reduxjs/toolkit'
import authReducer from './slices/authSlice'
import analyticsReducer from './slices/analyticsSlice'
import devicesReducer from './slices/devicesSlice'

export const store = configureStore({
  reducer: {
    auth: authReducer,
    analytics: analyticsReducer,
    devices: devicesReducer,
  },
})

