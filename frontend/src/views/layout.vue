<template>
  <div class="layout-shell">
    <!-- Sidebar -->
    <aside class="sidebar" :class="{ collapsed: isCollapse }">
      <div class="sidebar-brand" @click="$router.push('/dashboard')">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2" />
            <line x1="12" y1="22" x2="12" y2="15.5" />
            <polyline points="22 8.5 12 15.5 2 8.5" />
          </svg>
        </div>
        <span v-show="!isCollapse" class="brand-text">SafeScene</span>
      </div>

      <!-- Main Nav -->
      <nav class="sidebar-nav">
        <router-link to="/dashboard" class="nav-item" :class="{ active: $route.path === '/dashboard' }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <rect x="3" y="3" width="7" height="7" rx="1.5" />
            <rect x="14" y="3" width="7" height="7" rx="1.5" />
            <rect x="3" y="14" width="7" height="7" rx="1.5" />
            <rect x="14" y="14" width="7" height="7" rx="1.5" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">工作台</span>
        </router-link>

        <div class="nav-section-label" v-show="!isCollapse">场景图推理</div>

        <router-link to="/inference/pipeline" class="nav-item" :class="{ active: $route.path.startsWith('/inference') }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <circle cx="12" cy="12" r="3" />
            <circle cx="5" cy="5" r="2" />
            <circle cx="19" cy="5" r="2" />
            <circle cx="5" cy="19" r="2" />
            <circle cx="19" cy="19" r="2" />
            <line x1="7.2" y1="6.8" x2="9.5" y2="9.5" />
            <line x1="16.8" y1="6.8" x2="14.5" y2="9.5" />
            <line x1="7.2" y1="17.2" x2="9.5" y2="14.5" />
            <line x1="16.8" y1="17.2" x2="14.5" y2="14.5" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">推理流水线</span>
        </router-link>

        <router-link to="/rules" class="nav-item" :class="{ active: $route.path === '/rules' }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">危险规则库</span>
        </router-link>

        <div class="nav-section-label" v-show="!isCollapse">资源管理</div>

        <router-link to="/detection/dataset" class="nav-item" :class="{ active: $route.path.startsWith('/detection/dataset') }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">图像数据集</span>
        </router-link>

        <router-link to="/model/registry" class="nav-item" :class="{ active: $route.path.startsWith('/model') }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" />
            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
            <line x1="12" y1="22.08" x2="12" y2="12" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">模型注册</span>
        </router-link>

        <router-link to="/detection/task" class="nav-item" :class="{ active: $route.path === '/detection/task' }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">检测任务</span>
        </router-link>

        <router-link to="/detection/result" class="nav-item" :class="{ active: $route.path.startsWith('/detection/result') }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <path d="M21.21 15.89A10 10 0 1 1 8 2.83" />
            <path d="M22 12A10 10 0 0 0 12 2v10Z" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">检测结果</span>
        </router-link>

        <router-link to="/report" class="nav-item" :class="{ active: $route.path === '/report' }">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="nav-icon">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2Z" />
            <polyline points="14 2 14 8 20 8" />
          </svg>
          <span v-show="!isCollapse" class="nav-label">评估报告</span>
        </router-link>
      </nav>

      <!-- Bottom collapse toggle -->
      <button class="sidebar-toggle" @click="isCollapse = !isCollapse">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="nav-icon"
          :style="{ transform: isCollapse ? 'rotate(180deg)' : '' }">
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
    </aside>

    <!-- Main Area -->
    <div class="main-area">
      <!-- Header -->
      <header class="header">
        <div class="header-breadcrumb">
          <span class="bc-item">工作台</span>
          <template v-if="pageTitle !== '工作台'">
            <span class="bc-sep">/</span>
            <span class="bc-current">{{ pageTitle }}</span>
          </template>
        </div>

        <div class="header-actions">
          <!-- Big screen entry -->
          <router-link to="/screen" class="header-screen" title="智慧监控大屏">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="16" height="16">
              <rect x="2" y="3" width="20" height="14" rx="1.5" />
              <line x1="8" y1="21" x2="16" y2="21" />
              <line x1="12" y1="17" x2="12" y2="21" />
            </svg>
            <span>监控大屏</span>
          </router-link>

          <div class="header-divider"></div>

          <!-- Quick action: new pipeline -->
          <router-link to="/inference/pipeline" class="header-btn" title="新建推理">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="18" height="18">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          </router-link>

          <div class="header-divider"></div>

          <!-- User -->
          <el-dropdown trigger="click" @command="onUserCmd">
            <button class="user-btn">
              <div class="user-avatar">{{ userInitial }}</div>
              <span class="user-name">{{ userName }}</span>
            </button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="logout">退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </header>

      <!-- Content -->
      <main class="content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useUserStore } from '@/store/user'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const isCollapse = ref(false)

const pageTitle = computed(() => route.meta?.title || '工作台')
const userName = computed(() => userStore.userInfo?.realName || userStore.userInfo?.username || '用户')
const userInitial = computed(() => (userName.value || 'U')[0].toUpperCase())

function onUserCmd(cmd) {
  if (cmd === 'logout') {
    userStore.logout().then(() => router.push('/login'))
  }
}
</script>

<style scoped>
.layout-shell {
  display: flex;
  height: 100vh;
  background: var(--color-bg);
}

/* ── Sidebar ── */
.sidebar {
  width: var(--sidebar-width);
  display: flex;
  flex-direction: column;
  background: var(--color-sidebar);
  transition: width var(--transition-normal);
  overflow: hidden;
  flex-shrink: 0;
}
.sidebar.collapsed { width: var(--sidebar-collapsed); }

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 18px;
  cursor: pointer;
  color: #fff;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.brand-icon {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  background: var(--color-primary);
  border-radius: var(--radius-lg);
  color: #fff;
  flex-shrink: 0;
}
.brand-icon svg { width: 20px; height: 20px; }
.brand-text {
  font-size: var(--text-lg);
  font-weight: var(--font-bold);
  letter-spacing: -0.3px;
  white-space: nowrap;
}

/* ── Navigation ── */
.sidebar-nav {
  flex: 1;
  overflow-y: auto;
  padding: 10px 10px;
}

.nav-section-label {
  padding: 16px 10px 6px;
  font-size: var(--text-xs);
  font-weight: var(--font-semibold);
  color: rgba(255,255,255,0.25);
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 10px;
  height: 38px;
  border-radius: var(--radius-md);
  text-decoration: none;
  color: var(--color-sidebar-text);
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  transition: all var(--transition-fast);
  margin-bottom: 2px;
}
.nav-item:hover {
  background: var(--color-sidebar-hover);
  color: #e2e8f0;
}
.nav-item.active {
  background: var(--color-sidebar-active);
  color: var(--color-primary);
}
.nav-icon {
  width: 18px; height: 18px;
  flex-shrink: 0;
  opacity: 0.7;
}
.nav-item.active .nav-icon { opacity: 1; }
.nav-label { white-space: nowrap; }

/* ── Sidebar Toggle ── */
.sidebar-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 44px;
  border: none;
  border-top: 1px solid rgba(255,255,255,0.06);
  background: transparent;
  color: rgba(255,255,255,0.3);
  cursor: pointer;
  transition: color var(--transition-fast);
}
.sidebar-toggle:hover { color: rgba(255,255,255,0.6); }
.sidebar-toggle svg { transition: transform var(--transition-normal); }

/* ── Main Area ── */
.main-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

/* ── Header ── */
.header {
  height: var(--header-height);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.header-breadcrumb {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--text-sm);
}
.bc-item { color: var(--color-text-tertiary); }
.bc-sep { color: var(--color-slate-300); margin: 0 2px; }
.bc-current { color: var(--color-text-primary); font-weight: var(--font-medium); }

.header-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

.header-btn {
  width: 34px; height: 34px;
  display: flex; align-items: center; justify-content: center;
  border-radius: var(--radius-md);
  color: var(--color-text-secondary);
  background: transparent;
  border: none;
  cursor: pointer;
  transition: all var(--transition-fast);
  text-decoration: none;
}
.header-btn:hover { background: var(--color-slate-100); color: var(--color-text-primary); }

.header-screen {
  display: inline-flex; align-items: center; gap: 7px;
  height: 34px; padding: 0 14px;
  border-radius: var(--radius-md);
  color: #fff;
  background: linear-gradient(135deg, #f97316, #ea580c);
  border: none;
  font-size: 13px; font-weight: var(--font-semibold);
  letter-spacing: .02em;
  cursor: pointer;
  text-decoration: none;
  box-shadow: 0 2px 8px rgba(249, 115, 22, .28);
  transition: all var(--transition-fast);
}
.header-screen:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(249, 115, 22, .4); }
.header-screen svg { opacity: .95; }

.header-divider {
  width: 1px; height: 22px;
  background: var(--color-border);
  margin: 0 6px;
}

.user-btn {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 10px 4px 4px;
  border-radius: var(--radius-lg);
  border: none;
  background: transparent;
  cursor: pointer;
  transition: background var(--transition-fast);
}
.user-btn:hover { background: var(--color-slate-100); }

.user-avatar {
  width: 30px; height: 30px;
  border-radius: var(--radius-full);
  background: var(--color-primary);
  color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-size: var(--text-xs);
  font-weight: var(--font-bold);
}
.user-name { font-size: var(--text-sm); font-weight: var(--font-medium); color: var(--color-text-primary); }

/* ── Content ── */
.content {
  flex: 1;
  padding: 18px 22px;
  overflow-y: auto;
}
</style>
