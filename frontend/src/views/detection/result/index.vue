<template>
  <div class="result-list-page">
    <!-- Task Selector -->
    <el-card shadow="never" class="task-selector">
      <el-form :inline="true">
        <el-form-item label="选择任务">
          <el-select v-model="selectedTaskId" placeholder="选择已完成的检测任务" @change="onTaskChange" style="width:300px">
            <el-option v-for="t in completedTasks" :key="t.id" :label="`任务 #${t.id} (${t.createTime})`" :value="t.id" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchResults">查询</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Results Table -->
    <el-card shadow="never" style="margin-top:12px" v-if="selectedTaskId">
      <el-table :data="results" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="imageId" label="图片ID" width="80" />
        <el-table-column prop="primaryBehavior" label="主要行为" min-width="160">
          <template #default="{ row }">
            <el-tag :type="row.primaryBehavior === 'safe' ? 'success' : 'danger'" effect="dark">
              {{ behaviorLabel(row.primaryBehavior) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="primaryRisk" label="主要风险" width="140">
          <template #default="{ row }">
            <el-tag :type="row.primaryRisk === 'fall_risk' ? 'danger' : 'warning'" effect="plain">
              {{ riskLabel(row.primaryRisk) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="riskSeverity" label="风险等级" width="100" sortable>
          <template #default="{ row }">
            <el-progress :percentage="Math.round(row.riskSeverity * 100)" :stroke-width="8"
              :color="row.riskSeverity > 0.7 ? '#f56c6c' : row.riskSeverity > 0.4 ? '#e6a23c' : '#67c23a'" />
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100" sortable>
          <template #default="{ row }">
            <span :style="{ color: row.confidence > 0.7 ? '#67c23a' : '#e6a23c' }">
              {{ (row.confidence * 100).toFixed(1) }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button size="small" type="primary" link
              @click="$router.push(`/detection/result/${selectedTaskId}/${row.imageId}`)">
              详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        style="margin-top:16px;justify-content:flex-end"
        v-model:current-page="pageNum"
        v-model:page-size="pageSize"
        :total="total"
        layout="total, prev, pager, next"
        @change="fetchResults"
      />
    </el-card>

    <el-empty v-if="!selectedTaskId" description="请先选择一个已完成的检测任务" />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getDetectionTaskList, getDetectionResults } from '@/api/detection'

const BEHAVIOR_LABELS = {
  climbing_scaffold_frame: '攀爬脚手架框架', leaning_out: '身体探出',
  standing_on_guardrail: '站立在护栏上', throwing_material: '抛掷物料',
  leaning_outside_platform: '平台外探身', crossing_guardrail: '翻越护栏',
  working_outside_guardrail: '护栏外作业', climbing_cross_brace: '攀爬交叉支撑',
  unsafe_step_off: '不安全下步', missing_step: '踏空',
  unstable_posture: '不稳定姿势', throwing_objects: '抛物行为',
  general_work: '一般作业', rebar_work: '钢筋作业',
  mechanical_operation: '机械操作', excavation_proximity: '开挖临近',
  scaffold_work: '脚手架作业',
  safe: '安全', unknown: '未知',
}

const RISK_LABELS = {
  fall_risk: '坠落风险',
  struck_by_object_risk: '物体打击风险',
  collapse_risk: '坍塌风险',
  electrocution_risk: '触电风险',
  mechanical_risk: '机械伤害风险',
}

const completedTasks = ref([])
const selectedTaskId = ref(null)
const results = ref([])
const total = ref(0)
const pageNum = ref(1)
const pageSize = ref(20)
const loading = ref(false)

function behaviorLabel(k) { return BEHAVIOR_LABELS[k] || k }
function riskLabel(k) { return RISK_LABELS[k] || k }

async function fetchCompletedTasks() {
  const res = await getDetectionTaskList({ status: 'completed', pageSize: 100 })
  completedTasks.value = res.rows || []
}

function onTaskChange() {
  pageNum.value = 1
  fetchResults()
}

async function fetchResults() {
  if (!selectedTaskId.value) return
  loading.value = true
  try {
    const res = await getDetectionResults(selectedTaskId.value, {
      pageNum: pageNum.value, pageSize: pageSize.value,
    })
    results.value = res.rows
    total.value = res.total
  } finally {
    loading.value = false
  }
}

onMounted(fetchCompletedTasks)
</script>
