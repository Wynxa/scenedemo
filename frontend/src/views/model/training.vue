<template>
  <div class="model-training">
    <el-card shadow="never" class="toolbar">
      <el-button type="primary" @click="showCreateDialog = true">创建训练任务</el-button>
    </el-card>

    <!-- Training Task List -->
    <el-card shadow="never" style="margin-top:12px">
      <el-table :data="list" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="datasetId" label="数据集ID" width="100" />
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusTag(row.status)">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="指标" min-width="250">
          <template #default="{ row }">
            <template v-if="row.metricsJson?.behavior_f1">
              行为F1: {{ (row.metricsJson.behavior_f1 * 100).toFixed(1) }}% |
              风险F1: {{ ((row.metricsJson.risk_f1 || 0) * 100).toFixed(1) }}% |
              MAE: {{ row.metricsJson.severity_mae?.toFixed(4) }}
            </template>
            <span v-else style="color:#999">训练中...</span>
          </template>
        </el-table-column>
        <el-table-column prop="startTime" label="开始时间" width="170" />
        <el-table-column prop="endTime" label="结束时间" width="170" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="showTrainingDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination style="margin-top:16px;justify-content:flex-end"
        v-model:current-page="pageNum" v-model:page-size="pageSize" :total="total"
        layout="total, prev, pager, next" @change="fetchList" />
    </el-card>

    <!-- Create Training Dialog -->
    <el-dialog v-model="showCreateDialog" title="创建训练任务" width="600px">
      <el-form ref="trainFormRef" :model="trainForm" :rules="trainRules" label-width="120px">
        <el-form-item label="数据集" prop="datasetId">
          <el-select v-model="trainForm.datasetId" placeholder="选择数据集" style="width:100%">
            <el-option v-for="d in datasets" :key="d.id" :label="d.name" :value="d.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="隐藏维度">
          <el-input-number v-model="trainConfig.hiddenDim" :min="32" :max="512" :step="32" />
        </el-form-item>
        <el-form-item label="注意力头数">
          <el-input-number v-model="trainConfig.numHeads" :min="1" :max="16" />
        </el-form-item>
        <el-form-item label="层数">
          <el-input-number v-model="trainConfig.numLayers" :min="1" :max="8" />
        </el-form-item>
        <el-form-item label="Dropout">
          <el-input-number v-model="trainConfig.dropout" :min="0" :max="0.5" :step="0.05" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreateTraining">开始训练</el-button>
      </template>
    </el-dialog>

    <!-- Training Detail Dialog -->
    <el-dialog v-model="showDetailDialog" title="训练详情" width="500px">
      <pre class="json-block">{{ JSON.stringify(detailTask, null, 2) }}</pre>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getTrainingList, createTraining } from '@/api/model'
import { getDatasetList } from '@/api/dataset'

const loading = ref(false)
const list = ref([]); const total = ref(0)
const pageNum = ref(1); const pageSize = ref(10)

const showCreateDialog = ref(false)
const trainFormRef = ref(null)
const trainForm = reactive({ datasetId: null })
const trainRules = { datasetId: [{ required: true, message: '请选择数据集', trigger: 'change' }] }
const trainConfig = reactive({ hiddenDim: 128, numHeads: 4, numLayers: 2, dropout: 0.2 })
const datasets = ref([])

const showDetailDialog = ref(false)
const detailTask = ref({})

function statusTag(s) {
  const map = { pending: 'info', running: 'warning', completed: 'success', failed: 'danger' }
  return map[s] || 'info'
}
function statusLabel(s) {
  const map = { pending: '等待中', running: '运行中', completed: '已完成', failed: '失败' }
  return map[s] || s
}

async function fetchList() {
  loading.value = true
  try {
    const res = await getTrainingList({ pageNum: pageNum.value, pageSize: pageSize.value })
    list.value = res.rows; total.value = res.total
  } finally { loading.value = false }
}

async function handleCreateTraining() {
  const valid = await trainFormRef.value.validate().catch(() => false)
  if (!valid) return
  await createTraining({
    datasetId: trainForm.datasetId,
    configJson: { ...trainConfig },
  })
  ElMessage.success('训练任务已创建')
  showCreateDialog.value = false
  fetchList()
}

function showTrainingDetail(row) {
  detailTask.value = row
  showDetailDialog.value = true
}

onMounted(async () => {
  fetchList()
  const res = await getDatasetList({ pageSize: 100 })
  datasets.value = res.rows || []
})
</script>

<style scoped>
.json-block { background: #f5f5f5; padding: 12px; border-radius: 4px; font-size: 12px; white-space: pre-wrap; word-break: break-all; }
</style>
