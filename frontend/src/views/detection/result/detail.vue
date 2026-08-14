<template>
  <div class="result-detail" v-loading="loading">
    <el-page-header @back="$router.back()">
      <template #content>
        检测结果详情 — 任务 #{{ route.params.taskId }} / 图片 #{{ route.params.imageId }}
      </template>
    </el-page-header>

    <el-row :gutter="16" style="margin-top:16px">
      <!-- Left: Image with Worker Bboxes -->
      <el-col :span="14">
        <el-card shadow="hover" class="image-panel">
          <template #header>
            <span>检测画面</span>
            <el-tag style="float:right" :type="data.unsafeCount > 0 ? 'danger' : 'success'">
              {{ data.unsafeCount > 0 ? `${data.unsafeCount} 个不安全行为` : '全部安全' }}
            </el-tag>
          </template>
          <div class="image-container" ref="imageContainerRef">
            <!-- Placeholder canvas for annotation overlay -->
            <div class="annotation-viewport">
              <img v-if="imageUrl" :src="imageUrl" class="scene-image" ref="imageRef"
                @load="onImageLoad" />
              <div v-else class="no-image">
                <el-icon :size="64"><Picture /></el-icon>
                <p>{{ data.image?.fileName || '图片加载中...' }}</p>
              </div>
              <!-- Worker bounding boxes overlay -->
              <svg v-if="imageLoaded" class="bbox-overlay" :viewBox="`0 0 ${imgWidth} ${imgHeight}`">
                <g v-for="(det, idx) in data.detections" :key="idx">
                  <!-- Bbox rectangle -->
                  <rect
                    :x="bboxes[idx]?.x" :y="bboxes[idx]?.y"
                    :width="bboxes[idx]?.w" :height="bboxes[idx]?.h"
                    :fill="det.primaryBehavior === 'safe' ? 'rgba(103,194,58,0.15)' : 'rgba(245,108,108,0.2)'"
                    :stroke="selectedWorker === idx ? '#409eff' : (det.primaryBehavior === 'safe' ? '#67c23a' : '#f56c6c')"
                    :stroke-width="selectedWorker === idx ? 3 : 2"
                    style="cursor:pointer"
                    @click="selectWorker(idx)"
                  />
                  <!-- Worker label -->
                  <text
                    :x="(bboxes[idx]?.x || 0) + 4"
                    :y="(bboxes[idx]?.y || 0) - 6"
                    :fill="det.primaryBehavior === 'safe' ? '#67c23a' : '#f56c6c'"
                    font-size="12" font-weight="bold"
                    style="cursor:pointer"
                    @click="selectWorker(idx)"
                  >
                    Worker {{ idx + 1 }} — {{ det.primaryBehaviorLabel }}
                  </text>
                </g>
              </svg>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- Right: Detection Details -->
      <el-col :span="10">
        <!-- Worker List -->
        <el-card shadow="hover" class="worker-list-card">
          <template #header><span>工人检测列表 ({{ data.totalWorkers || 0 }})</span></template>
          <div class="worker-list">
            <div v-for="(det, idx) in data.detections" :key="idx"
              class="worker-item"
              :class="{ active: selectedWorker === idx }"
              @click="selectWorker(idx)">
              <div class="worker-header">
                <span class="worker-name">工人 #{{ idx + 1 }}</span>
                <el-tag size="small" :type="det.primaryBehavior === 'safe' ? 'success' : 'danger'">
                  {{ det.primaryBehaviorLabel }}
                </el-tag>
              </div>
              <div class="worker-meta">
                <span>置信度: {{ (det.confidence * 100).toFixed(1) }}%</span>
                <span>风险: {{ det.riskSeverity.toFixed(2) }}</span>
              </div>
              <el-progress
                :percentage="Math.round(det.riskSeverity * 100)"
                :stroke-width="4"
                :color="det.riskSeverity > 0.7 ? '#f56c6c' : det.riskSeverity > 0.4 ? '#e6a23c' : '#67c23a'"
              />
            </div>
            <el-empty v-if="!data.detections?.length" description="未检测到工人" :image-size="48" />
          </div>
        </el-card>

        <!-- Behavior Probability Chart -->
        <el-card v-if="selectedDet" shadow="hover" style="margin-top:12px">
          <template #header><span>行为概率分布 — 工人 #{{ selectedWorker + 1 }}</span></template>
          <div ref="behaviorChartRef" style="height:220px"></div>
        </el-card>

        <!-- Risk Radar Chart -->
        <el-card v-if="selectedDet" shadow="hover" style="margin-top:12px">
          <template #header><span>风险评估</span></template>
          <div style="display:flex;align-items:center;gap:20px">
            <div ref="riskRadarRef" style="width:160px;height:160px"></div>
            <div style="flex:1">
              <div class="risk-item" v-for="(prob, riskKey) in selectedDet.riskProbs" :key="riskKey">
                <span class="risk-label">{{ riskLabels[riskKey] || riskKey }}</span>
                <el-progress :percentage="Math.round(prob * 100)" :stroke-width="10"
                  :color="prob > 0.6 ? '#f56c6c' : '#e6a23c'" />
                <span class="risk-value">{{ (prob * 100).toFixed(1) }}%</span>
              </div>
              <div class="severity-gauge">
                <span>综合风险等级</span>
                <span class="severity-value" :style="{ color: selectedDet.riskSeverity > 0.7 ? '#f56c6c' : '#e6a23c' }">
                  {{ (selectedDet.riskSeverity * 100).toFixed(1) }}%
                </span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { Picture } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import { getResultDetail } from '@/api/detection'

const route = useRoute()
const loading = ref(false)
const data = ref({ detections: [], image: null, totalWorkers: 0, unsafeCount: 0 })
const selectedWorker = ref(0)

const imageRef = ref(null)
const imageContainerRef = ref(null)
const imgWidth = ref(800)
const imgHeight = ref(600)
const imageLoaded = ref(false)

const behaviorChartRef = ref(null)
const riskRadarRef = ref(null)
let behaviorChart = null
let riskRadar = null

const riskLabels = {
  fall_risk: '坠落风险', struck_by_object_risk: '物体打击风险',
  collapse_risk: '坍塌风险', electrocution_risk: '触电风险',
  mechanical_risk: '机械伤害风险',
}

const imageUrl = computed(() => {
  if (data.value.image?.filePath) return `/uploads/${data.value.image.filePath}`
  return null
})

const selectedDet = computed(() => data.value.detections?.[selectedWorker.value])

const bboxes = computed(() => {
  return (data.value.detections || []).map(d => {
    const parts = (d.workerBbox || '0,0,0,0').split(',').map(Number)
    return { x: parts[0], y: parts[1], w: parts[2], h: parts[3] }
  })
})

function selectWorker(idx) {
  selectedWorker.value = idx
  nextTick(() => {
    updateBehaviorChart()
    updateRiskRadar()
  })
}

function onImageLoad() {
  const img = imageRef.value
  if (img) {
    imgWidth.value = img.naturalWidth || 800
    imgHeight.value = img.naturalHeight || 600
    imageLoaded.value = true
  }
}

function updateBehaviorChart() {
  if (!behaviorChartRef.value || !selectedDet.value) return
  if (!behaviorChart) behaviorChart = echarts.init(behaviorChartRef.value)

  const probs = selectedDet.value.behaviorProbs || {}
  const entries = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)

  behaviorChart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 120, right: 40, top: 10, bottom: 20 },
    xAxis: { type: 'value', max: 1, axisLabel: { formatter: v => (v * 100).toFixed(0) + '%' } },
    yAxis: { type: 'category', data: entries.map(e => e[0]), inverse: true,
      axisLabel: { fontSize: 10, width: 110, overflow: 'truncate' } },
    series: [{
      type: 'bar',
      data: entries.map(e => ({
        value: e[1],
        itemStyle: { color: e[1] > 0.5 ? '#f56c6c' : e[1] > 0.3 ? '#e6a23c' : '#409eff', borderRadius: [0, 4, 4, 0] },
      })),
      barMaxWidth: 20,
    }],
  }, true)
}

function updateRiskRadar() {
  if (!riskRadarRef.value || !selectedDet.value) return
  if (!riskRadar) riskRadar = echarts.init(riskRadarRef.value)

  const probs = selectedDet.value.riskProbs || {}
  riskRadar.setOption({
    tooltip: {},
    radar: {
      center: ['50%', '50%'],
      radius: '70%',
      indicator: Object.keys(probs).map(k => ({ name: riskLabels[k] || k, max: 1 })),
    },
    series: [{
      type: 'radar',
      data: [{ value: Object.values(probs), name: '风险概率',
        areaStyle: { color: 'rgba(245,108,108,0.3)' } }],
    }],
  }, true)
}

watch(selectedDet, () => {
  nextTick(() => { updateBehaviorChart(); updateRiskRadar() })
})

onMounted(async () => {
  loading.value = true
  try {
    data.value = await getResultDetail(route.params.taskId, route.params.imageId)
    if (data.value.detections?.length) {
      nextTick(() => { updateBehaviorChart(); updateRiskRadar() })
    }
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.result-detail { }
.image-panel { }
.image-container { position: relative; }
.annotation-viewport { position: relative; width: 100%; }
.scene-image { width: 100%; display: block; border-radius: 4px; }
.bbox-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
.bbox-overlay rect, .bbox-overlay text { pointer-events: auto; }
.no-image { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px; background: #f5f5f5; border-radius: 4px; color: #999; }
.worker-list-card { max-height: 360px; }
.worker-list { max-height: 300px; overflow-y: auto; }
.worker-item {
  padding: 10px 12px; border: 1px solid #eee; border-radius: 8px; margin-bottom: 8px; cursor: pointer; transition: all 0.2s;
}
.worker-item:hover { border-color: #409eff; }
.worker-item.active { border-color: #409eff; background: #ecf5ff; }
.worker-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.worker-name { font-weight: 600; font-size: 14px; }
.worker-meta { display: flex; gap: 16px; font-size: 12px; color: #999; margin-bottom: 6px; }
.risk-item { margin-bottom: 10px; }
.risk-label { font-size: 13px; color: #666; display: block; margin-bottom: 4px; }
.risk-value { font-size: 13px; font-weight: 600; color: #333; }
.severity-gauge { display: flex; justify-content: space-between; align-items: center; margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee; }
.severity-value { font-size: 24px; font-weight: 700; }
</style>
