<template>
  <div class="model-compare">
    <el-card shadow="never">
      <el-row :gutter="16" align="middle">
        <el-col :span="10">
          <el-select v-model="modelAId" placeholder="选择模型A" @change="fetchCompare" style="width:100%">
            <el-option v-for="m in models" :key="m.id" :label="`${m.name} v${m.version}`" :value="m.id" />
          </el-select>
        </el-col>
        <el-col :span="4" style="text-align:center">
          <span style="font-size:20px;color:#999">VS</span>
        </el-col>
        <el-col :span="10">
          <el-select v-model="modelBId" placeholder="选择模型B" @change="fetchCompare" style="width:100%">
            <el-option v-for="m in models" :key="m.id" :label="`${m.name} v${m.version}`" :value="m.id" />
          </el-select>
        </el-col>
      </el-row>
    </el-card>

    <!-- Comparison Results -->
    <el-row :gutter="16" style="margin-top:16px" v-if="comparison">
      <!-- Metrics Table -->
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>指标对比</template>
          <el-table :data="diffTable" stripe>
            <el-table-column prop="metric" label="指标" width="140" />
            <el-table-column prop="valueA" label="模型A" width="120" />
            <el-table-column prop="valueB" label="模型B" width="120" />
            <el-table-column prop="delta" label="差异" width="100">
              <template #default="{ row }">
                <span :style="{ color: row.delta > 0 ? '#67c23a' : row.delta < 0 ? '#f56c6c' : '#999' }">
                  {{ row.delta ? (row.delta > 0 ? '+' : '') + row.delta : '-' }}
                </span>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>

      <!-- Chart -->
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>指标差值可视化</template>
          <div ref="diffChartRef" style="height:300px"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-empty v-if="!comparison && (modelAId || modelBId)" description="请选择两个模型进行对比" />
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { getModelList } from '@/api/model'

const models = ref([])
const modelAId = ref(null)
const modelBId = ref(null)
const comparison = ref(null)
const diffChartRef = ref(null)
let diffChart = null

const diffTable = computed(() => {
  if (!comparison.value) return []
  const { comparison: comp } = comparison.value
  return Object.entries(comp).map(([k, v]) => ({
    metric: k,
    valueA: v.modelA,
    valueB: v.modelB,
    delta: v.delta,
  }))
})

watch([modelAId, modelBId], () => {
  if (modelAId.value && modelBId.value) fetchCompare()
})

async function fetchCompare() {
  if (!modelAId.value || !modelBId.value) return
  // Client-side comparison: get both models and compute diff
  const all = await getModelList({ pageSize: 100 })
  const a = (all.rows || []).find(m => m.id === modelAId.value)
  const b = (all.rows || []).find(m => m.id === modelBId.value)
  if (!a || !b) return

  const metricsA = a.metricsJson || {}
  const metricsB = b.metricsJson || {}
  const comp = {}
  const allKeys = new Set([...Object.keys(metricsA), ...Object.keys(metricsB)])
  allKeys.forEach(k => {
    comp[k] = {
      modelA: metricsA[k] ?? '-',
      modelB: metricsB[k] ?? '-',
      delta: (typeof metricsA[k] === 'number' && typeof metricsB[k] === 'number')
        ? +(metricsB[k] - metricsA[k]).toFixed(4) : null,
    }
  })

  comparison.value = { modelA: { id: a.id, name: a.name }, modelB: { id: b.id, name: b.name }, comparison: comp }

  nextTick(updateDiffChart)
}

function updateDiffChart() {
  if (!diffChartRef.value) return
  if (!diffChart) diffChart = echarts.init(diffChartRef.value)

  const comp = comparison.value?.comparison || {}
  const entries = Object.entries(comp).filter(([, v]) => v.delta !== null)

  diffChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: 120, right: 40, top: 10, bottom: 20 },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: entries.map(e => e[0]) },
    series: [{
      type: 'bar',
      data: entries.map(e => ({
        value: e[1].delta,
        itemStyle: { color: e[1].delta > 0 ? '#67c23a' : '#f56c6c' },
      })),
    }],
  }, true)
}

onMounted(async () => {
  const res = await getModelList({ pageSize: 100 })
  models.value = res.rows || []
})
</script>
