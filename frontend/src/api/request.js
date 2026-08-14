import axios from 'axios'
import { ElMessage } from 'element-plus'

const service = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// Request interceptor: attach token
service.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor: unwrap RuoYi format {code, msg, data}
service.interceptors.response.use(
  (response) => {
    const { code, msg, data, rows, total } = response.data
    if (code === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
      return Promise.reject(new Error(msg || 'Unauthorized'))
    }
    if (code !== 200) {
      ElMessage.error(msg || '请求失败')
      return Promise.reject(new Error(msg || 'Error'))
    }
    // For table data, return {rows, total}
    if (rows !== undefined) {
      return { rows, total }
    }
    return data
  },
  (error) => {
    ElMessage.error(error.message || '网络错误')
    return Promise.reject(error)
  }
)

export default service
