import { defineStore } from 'pinia'
import { login as loginApi, logout as logoutApi, getUserInfo } from '@/api/auth'

export const useUserStore = defineStore('user', {
  state: () => ({
    token: localStorage.getItem('token') || '',
    userInfo: null,
    roles: [],
    permissions: [],
  }),

  actions: {
    async login(username, password) {
      const data = await loginApi(username, password)
      this.token = data.token
      this.userInfo = data.user
      localStorage.setItem('token', data.token)
      return data
    },

    async getUserInfo() {
      const data = await getUserInfo()
      this.userInfo = data.user
      this.roles = data.roles || []
      this.permissions = data.permissions || []
      return data
    },

    async logout() {
      await logoutApi()
      this.token = ''
      this.userInfo = null
      localStorage.removeItem('token')
    },

    resetToken() {
      this.token = ''
      this.userInfo = null
      localStorage.removeItem('token')
    },
  },
})
