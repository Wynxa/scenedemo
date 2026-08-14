<template>
  <div class="report-detail" v-loading="loading">
    <!-- Header -->
    <div class="report-header">
      <el-page-header @back="$router.back()">
        <template #content>
          <span class="header-title">评估报告 #{{ reportId }}</span>
        </template>
      </el-page-header>
      <div class="header-meta" v-if="report">
        <span>生成时间：{{ report.createTime }}</span>
        <el-tag v-if="task" :type="task.status === 'completed' ? 'success' : 'info'" size="small">
          {{ task.status === 'completed' ? '已完成' : task.status }}
        </el-tag>
      </div>
    </div>

    <template v-if="report && task">
      <!-- Stats Overview -->
      <el-row :gutter="16" class="stats-row">
        <el-col :span="6">
          <div class="stat-card">
            <div class="stat-value">{{ totalWorkers }}</div>
            <div class="stat-label">总检测工人</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-card unsafe">
            <div class="stat-value">{{ unsafeCount }}</div>
            <div class="stat-label">不安全行为</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-card">
            <div class="stat-value">{{ unsafeRate }}%</div>
            <div class="stat-label">不安全率</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-card risk">
            <div class="stat-value">{{ highestRisk || '无' }}</div>
            <div class="stat-label">最高风险类别</div>
          </div>
        </el-col>
      </el-row>

      <!-- Section 1: Original & Detection Images -->
      <el-card shadow="never" class="section-card">
        <template #header><h3>图像检测结果</h3></template>
        <el-row :gutter="16">
          <el-col :span="12">
            <div class="image-panel">
              <h4 class="panel-title">原图</h4>
              <img v-if="imageUrl" :src="imageUrl" class="scene-image" />
              <el-empty v-else description="无图像" :image-size="80" />
            </div>
          </el-col>
          <el-col :span="12">
            <div class="image-panel">
              <h4 class="panel-title">检测标注</h4>
              <div class="annotation-viewport" v-if="imageUrl">
                <img :src="imageUrl" class="scene-image" ref="detImgRef" @load="onDetImgLoad" />
                <svg v-if="imgLoaded" class="bbox-overlay" :viewBox="`0 0 ${imgW} ${imgH}`">
                  <g v-for="(obj, idx) in workerObjects" :key="idx">
                    <rect
                      :x="obj.bbox[0]" :y="obj.bbox[1]"
                      :width="obj.bbox[2] - obj.bbox[0]"
                      :height="obj.bbox[3] - obj.bbox[1]"
                      :fill="workerFill(idx)"
                      :stroke="workerStroke(idx)"
                      stroke-width="2"
                    />
                    <text
                      :x="obj.bbox[0] + 4"
                      :y="obj.bbox[1] - 6"
                      :fill="workerStroke(idx)"
                      font-size="12" font-weight="bold"
                    >
                      W{{ idx + 1 }}
                    </text>
                  </g>
                </svg>
              </div>
              <el-empty v-else description="无图像" :image-size="80" />
            </div>
          </el-col>
        </el-row>
      </el-card>

      <!-- Section 2: Scene Graph -->
      <el-card shadow="never" class="section-card">
        <template #header><h3>场景图</h3></template>
        <SceneGraphView
          v-if="objects.length"
          :objects="objects"
          :relations="relations"
          :hazards="hazards"
          :image-width="imgW || 800"
          :image-height="imgH || 600"
          :image-url="imageUrl"
        />
        <el-empty v-else description="无场景图数据" :image-size="60" />
      </el-card>

      <!-- Section 3: Hazard Assessment -->
      <el-card shadow="never" class="section-card">
        <template #header><h3>风险研判结果</h3></template>
        <HazardResultsPanel v-if="hazards.length" :worker-results="hazards" />
        <el-empty v-else description="无风险研判数据" :image-size="60" />
      </el-card>

      <!-- Section 4: Conclusion -->
      <el-card shadow="never" class="section-card conclusion-card">
        <template #header><h3>综合结论</h3></template>
        <div class="conclusion-body">
          <div class="conclusion-status" :class="{ unsafe: summary?.hasUnsafeBehavior }">
            <el-icon :size="28">
              <CircleCheckFilled v-if="!summary?.hasUnsafeBehavior" />
              <WarningFilled v-else />
            </el-icon>
            <span class="conclusion-text">
              {{ summary?.hasUnsafeBehavior ? '该施工现场存在不安全行为，需要立即整改' : '该施工现场未发现明显不安全行为' }}
            </span>
          </div>

          <el-descriptions :column="2" border class="conclusion-details" v-if="summary">
            <el-descriptions-item label="不安全工人数">
              {{ summary.unsafeWorkerCount ?? 0 }}
            </el-descriptions-item>
            <el-descriptions-item label="最高风险类别">
              <el-tag type="danger" size="small">{{ summary.highestRiskCategory || '无' }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="最高风险评分">
              {{ ((summary.highestRiskScore ?? 0) * 100).toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="任务流水线">
              {{ pipelineLabel(task.pipelineType) }}
            </el-descriptions-item>
          </el-descriptions>

          <div class="ppe-summary" v-if="ppeAlerts.length">
            <h4>PPE 缺失汇总</h4>
            <el-tag v-for="(alert, i) in ppeAlerts" :key="i" type="danger" effect="dark" style="margin:4px">
              {{ alert }}
            </el-tag>
          </div>
        </div>
      </el-card>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { CircleCheckFilled, WarningFilled } from '@element-plus/icons-vue'
import { getReport } from '@/api/report'
import SceneGraphView from '@/views/inference/components/SceneGraphView.vue'
import HazardResultsPanel from '@/views/inference/components/HazardResultsPanel.vue'

const route = useRoute()
const reportId = computed(() => route.params.id)
const loading = ref(false)
const report = ref(null)
const task = ref(null)
const image = ref(null)
const objects = ref([])
const relations = ref([])
const hazards = ref([])
const summary = ref(null)

const detImgRef = ref(null)
const imgW = ref(800)
const imgH = ref(600)
const imgLoaded = ref(false)

const imageUrl = computed(() => {
  if (image.value?.filePath) return `/uploads/${image.value.filePath}`
  return null
})

const workerObjects = computed(() =>
  objects.value.filter(o => o.labelName && /worker|person|工人|人员/i.test(o.labelName))
)

const totalWorkers = computed(() => workerObjects.value.length)
const unsafeCount = computed(() =>
  hazards.value.filter(h => h.unsafeBehaviorCategory && h.unsafeBehaviorCategory !== 'safe').length
)
const unsafeRate = computed(() =>
  totalWorkers.value ? Math.round((unsafeCount.value / totalWorkers.value) * 100) : 0
)
const highestRisk = computed(() => summary.value?.highestRiskCategory || null)

const ppeAlerts = computed(() => {
  const alerts = []
  for (const h of hazards.value) {
    const missing = h.missingPpeJson || []
    for (const item of missing) {
      const msg = `工人缺失: ${item}`
      if (!alerts.includes(msg)) alerts.push(msg)
    }
  }
  return alerts
})

function pipelineLabel(t) {
  const map = { full_pipeline: '全流程推理', scene_graph_only: '场景图推理', reasoning_only: '风险推理', detector_only: '目标检测' }
  return map[t] || t
}

function workerFill(idx) {
  const h = hazards.value[idx]
  if (h && h.unsafeBehaviorCategory && h.unsafeBehaviorCategory !== 'safe') return 'rgba(245,108,108,0.2)'
  return 'rgba(103,194,58,0.15)'
}

function workerStroke(idx) {
  const h = hazards.value[idx]
  if (h && h.unsafeBehaviorCategory && h.unsafeBehaviorCategory !== 'safe') return '#f56c6c'
  return '#67c23a'
}

function onDetImgLoad() {
  const img = detImgRef.value
  if (img) {
    imgW.value = img.naturalWidth || 800
    imgH.value = img.naturalHeight || 600
    imgLoaded.value = true
  }
}

onMounted(async () => {
  loading.value = true
  try {
    const data = await getReport(reportId.value)
    report.value = data.report
    task.value = data.task
    image.value = data.image
    objects.value = data.objects || []
    relations.value = data.relations || []
    hazards.value = data.hazards || []
    summary.value = data.summary
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.report-detail {
  max-width: 1200px;
  margin: 0 auto;
  padding-bottom: 40px;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.header-title {
  font-size: 18px;
  font-weight: 700;
  color: #1e293b;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  color: #64748b;
}

.stats-row {
  margin-bottom: 16px;
}

.stat-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
}

.stat-card.unsafe {
  border-left: 4px solid #f56c6c;
}

.stat-card.risk {
  border-left: 4px solid #e6a23c;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  color: #1e293b;
}

.stat-card.unsafe .stat-value {
  color: #f56c6c;
}

.stat-card.risk .stat-value {
  color: #e6a23c;
}

.stat-label {
  font-size: 12px;
  color: #94a3b8;
  margin-top: 4px;
}

.section-card {
  margin-bottom: 16px;
}

.section-card h3 {
  margin: 0;
  font-size: 15px;
  color: #1e293b;
}

.image-panel {
  text-align: center;
}

.panel-title {
  margin: 0 0 8px 0;
  font-size: 13px;
  color: #64748b;
  font-weight: 500;
}

.scene-image {
  width: 100%;
  border-radius: 4px;
  display: block;
}

.annotation-viewport {
  position: relative;
  width: 100%;
}

.bbox-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.conclusion-body {
  padding: 8px 0;
}

.conclusion-status {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px;
  border-radius: 8px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  margin-bottom: 20px;
}

.conclusion-status.unsafe {
  background: #fef2f2;
  border-color: #fecaca;
}

.conclusion-status .el-icon {
  color: #22c55e;
}

.conclusion-status.unsafe .el-icon {
  color: #ef4444;
}

.conclusion-text {
  font-size: 16px;
  font-weight: 600;
  color: #166534;
}

.conclusion-status.unsafe .conclusion-text {
  color: #991b1b;
}

.conclusion-details {
  margin-bottom: 16px;
}

.ppe-summary {
  margin-top: 12px;
}

.ppe-summary h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #475569;
}
</style>
