import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/login.vue'),
    meta: { title: '登录' },
  },
  {
    path: '/screen',
    name: 'Screen',
    component: () => import('@/views/screen/index.vue'),
    meta: { title: '智慧监控大屏' },
  },
  {
    path: '/',
    component: () => import('@/views/layout.vue'),
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/dashboard/index.vue'),
        meta: { title: '工作台', icon: 'HomeFilled' },
      },
      {
        path: 'detection/dataset',
        name: 'DatasetList',
        component: () => import('@/views/detection/dataset/index.vue'),
        meta: { title: '图像数据集', icon: 'Picture' },
      },
      {
        path: 'detection/dataset/:id',
        name: 'DatasetDetail',
        component: () => import('@/views/detection/dataset/detail.vue'),
        meta: { title: '数据集详情', hidden: true },
      },
      {
        path: 'detection/task',
        name: 'DetectionTask',
        component: () => import('@/views/detection/task/index.vue'),
        meta: { title: '检测任务', icon: 'VideoCamera' },
      },
      {
        path: 'detection/result',
        name: 'DetectionResult',
        component: () => import('@/views/detection/result/index.vue'),
        meta: { title: '检测结果', icon: 'DataAnalysis' },
      },
      {
        path: 'detection/result/:taskId/:imageId',
        name: 'ResultDetail',
        component: () => import('@/views/detection/result/detail.vue'),
        meta: { title: '结果详情', hidden: true },
      },
      {
        path: 'model/registry',
        name: 'ModelRegistry',
        component: () => import('@/views/model/registry.vue'),
        meta: { title: '模型注册', icon: 'Cpu' },
      },
      {
        path: 'model/training',
        name: 'ModelTraining',
        component: () => import('@/views/model/training.vue'),
        meta: { title: '模型训练', icon: 'DataLine' },
      },
      {
        path: 'model/compare',
        name: 'ModelCompare',
        component: () => import('@/views/model/compare.vue'),
        meta: { title: '模型对比', icon: 'Switch' },
      },
      {
        path: 'report',
        name: 'ReportList',
        component: () => import('@/views/report/list.vue'),
        meta: { title: '评估报告', icon: 'Document' },
      },
      {
        path: 'report/:id',
        name: 'ReportDetail',
        component: () => import('@/views/report/detail.vue'),
        meta: { title: '报告详情', hidden: true },
      },
      {
        path: 'inference/pipeline',
        name: 'InferencePipeline',
        component: () => import('@/views/inference/pipeline.vue'),
        meta: { title: '场景图推理', icon: 'Share' },
      },
      {
        path: 'rules',
        name: 'HazardRules',
        component: () => import('@/views/rules/index.vue'),
        meta: { title: '危险规则库', icon: 'List' },
      },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// Navigation guard
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (to.path !== '/login' && !token) {
    next('/login')
  } else if (to.path === '/login' && token) {
    next('/dashboard')
  } else {
    next()
  }
})

export default router
