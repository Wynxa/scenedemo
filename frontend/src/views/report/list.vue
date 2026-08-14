<template>
  <div class="report-page">
    <el-card shadow="never" class="toolbar">
      <el-button type="primary" @click="showGenerateDialog = true">生成报告</el-button>
    </el-card>

    <el-card shadow="never" style="margin-top:12px">
      <el-table :data="list" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="reportType" label="类型" width="120">
          <template #default="{ row }">
            <el-tag>{{ reportTypeLabel(row.reportType) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="taskId" label="关联任务" width="100" />
        <el-table-column prop="createTime" label="生成时间" width="170" />
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="viewReport(row)">查看</el-button>
            <el-button size="small" type="success" link @click="downloadReport(row)"
              :disabled="!row.filePath">
              下载
            </el-button>
            <el-button size="small" type="danger" link @click="handleDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination style="margin-top:16px;justify-content:flex-end"
        v-model:current-page="pageNum" v-model:page-size="pageSize" :total="total"
        layout="total, prev, pager, next" @change="fetchList" />
    </el-card>

    <!-- Generate Dialog -->
    <el-dialog v-model="showGenerateDialog" title="生成评估报告" width="480px">
      <el-form label-width="100px">
        <el-form-item label="检测任务">
          <el-select v-model="genTaskId" placeholder="选择已完成的任务" style="width:100%">
            <el-option v-for="t in completedTasks" :key="t.id" :label="`任务 #${t.id}`" :value="t.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="报告类型">
          <el-select v-model="genType" style="width:100%">
            <el-option label="评估报告" value="evaluation" />
            <el-option label="模型对比报告" value="comparison" />
            <el-option label="消融实验报告" value="ablation" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showGenerateDialog = false">取消</el-button>
        <el-button type="primary" @click="handleGenerate">生成</el-button>
      </template>
    </el-dialog>

  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getReportList, generateReport, deleteReport } from '@/api/report'
import { getDetectionTaskList } from '@/api/detection'

const router = useRouter()
const loading = ref(false)
const list = ref([]); const total = ref(0)
const pageNum = ref(1); const pageSize = ref(10)

const showGenerateDialog = ref(false)
const genTaskId = ref(null); const genType = ref('evaluation')
const completedTasks = ref([])

function reportTypeLabel(t) {
  const map = { evaluation: '评估报告', comparison: '对比报告', ablation: '消融报告' }
  return map[t] || t
}

async function fetchList() {
  loading.value = true
  try {
    const res = await getReportList({ pageNum: pageNum.value, pageSize: pageSize.value })
    list.value = res.rows; total.value = res.total
  } finally { loading.value = false }
}

async function handleGenerate() {
  await generateReport({ taskId: genTaskId.value, reportType: genType.value })
  ElMessage.success('报告生成任务已创建')
  showGenerateDialog.value = false
  fetchList()
}

function viewReport(row) {
  router.push(`/report/${row.id}`)
}

function downloadReport(row) {
  window.open(`/api/report/${row.id}/download`, '_blank')
}

async function handleDelete(row) {
  await ElMessageBox.confirm('确定删除该报告？', '警告', { type: 'warning' })
  await deleteReport(row.id)
  ElMessage.success('删除成功')
  fetchList()
}

onMounted(async () => {
  fetchList()
  const tasksRes = await getDetectionTaskList({ status: 'completed', pageSize: 100 })
  completedTasks.value = tasksRes.rows || []
})
</script>

<style scoped>
</style>
