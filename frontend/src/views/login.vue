<template>
  <div class="login-page">
    <!-- Background pattern -->
    <div class="login-bg">
      <div class="bg-grid"></div>
      <div class="bg-gradient"></div>
    </div>

    <!-- Card -->
    <div class="login-card">
      <div class="login-brand">
        <div class="brand-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2" />
            <line x1="12" y1="22" x2="12" y2="15.5" />
            <polyline points="22 8.5 12 15.5 2 8.5" />
          </svg>
        </div>
        <h1 class="brand-name">SafeScene</h1>
        <p class="brand-desc">施工场景安全智能分析平台</p>
      </div>

      <el-form ref="formRef" :model="form" :rules="rules" size="large" class="login-form">
        <el-form-item prop="username">
          <el-input
            v-model="form.username"
            placeholder="用户名"
            :prefix-icon="UserFilled"
            class="login-input"
          />
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="密码"
            :prefix-icon="Lock"
            show-password
            class="login-input"
            @keyup.enter="handleLogin"
          />
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="loading"
            @click="handleLogin"
            class="login-btn"
          >
            登 录
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { UserFilled, Lock } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useUserStore } from '@/store/user'

const router = useRouter()
const userStore = useUserStore()
const loading = ref(false)
const formRef = ref(null)

const form = reactive({ username: 'admin', password: 'admin123' })

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

async function handleLogin() {
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return
  loading.value = true
  try {
    await userStore.login(form.username, form.password)
    ElMessage.success('登录成功')
    router.push('/dashboard')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  overflow: hidden;
}

/* ── Background ── */
.login-bg {
  position: absolute;
  inset: 0;
}
.bg-grid {
  position: absolute;
  inset: 0;
  background:
    linear-gradient(rgba(15, 23, 42, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15, 23, 42, 0.05) 1px, transparent 1px);
  background-size: 60px 60px;
}
.bg-gradient {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 50% 40%, rgba(249, 115, 22, 0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 80% 20%, rgba(37, 99, 235, 0.04) 0%, transparent 50%);
}

/* ── Card ── */
.login-card {
  position: relative;
  width: 400px;
  padding: 44px 40px 36px;
  background: #fff;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-xl);
}

.login-brand {
  text-align: center;
  margin-bottom: 32px;
}

.brand-mark {
  width: 48px; height: 48px;
  margin: 0 auto 14px;
  display: flex; align-items: center; justify-content: center;
  background: var(--color-primary);
  border-radius: var(--radius-xl);
  color: #fff;
}
.brand-mark svg { width: 28px; height: 28px; }

.brand-name {
  font-size: var(--text-3xl);
  font-weight: var(--font-bold);
  color: var(--color-slate-900);
  letter-spacing: -0.5px;
  margin: 0 0 4px;
}
.brand-desc {
  font-size: var(--text-sm);
  color: var(--color-text-tertiary);
  margin: 0;
}

/* ── Form ── */
.login-form {
  margin-top: 0;
}

:deep(.login-input .el-input__wrapper) {
  background: var(--color-slate-50);
  border: 1px solid var(--color-border);
  box-shadow: none;
  border-radius: var(--radius-md);
  transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
}
:deep(.login-input .el-input__wrapper:hover) {
  border-color: var(--color-slate-300);
}
:deep(.login-input .el-input__wrapper.is-focus) {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.1);
}

.login-btn {
  width: 100%;
  height: 44px;
  font-size: var(--text-base);
  font-weight: var(--font-semibold);
  letter-spacing: 0.5px;
  border-radius: var(--radius-md);
  margin-top: 4px;
}
</style>
