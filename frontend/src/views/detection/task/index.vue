<template>
  <div class="task-page">
    <el-card shadow="never">
      <el-form :inline="true">
        <el-form-item>
          <el-select v-model="filterStatus" placeholder="状态筛选" clearable @change="fetchList">
            <el-option label="等待中" value="pending" />
            <el-option label="运行中" value="running" />
            <el-option label="已完成" value="completed" />
            <el-option label="失败" value="failed" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchList">刷新</el-button>
          <el-button type="success" @click="openCreateDialog">新建任务</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" style="margin-top:12px">
      <el-table :data="list" stripe v-loading="loading">
        <el-table-column prop="id" label="任务ID" width="80" />
        <el-table-column prop="taskName" label="任务名称" min-width="160">
          <template #default="{ row }">
            <span>{{ row.taskName || `推理任务 #${row.id}` }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="pipelineType" label="流水线类型" width="130">
          <template #default="{ row }">
            <el-tag :type="pipelineTag(row.pipelineType)" size="small">
              {{ pipelineLabel(row.pipelineType) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="imageId" label="图像ID" width="80" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="statusTag(row.status)">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="创建时间" width="170" />
        <el-table-column prop="finishTime" label="完成时间" width="170" />
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button v-if="row.status === 'completed'" size="small" type="primary" link
              @click="viewResults(row)">
              查看结果
            </el-button>
            <el-button size="small" type="danger" link
              @click="handleDelete(row)"
              :disabled="row.status === 'running'">
              删除
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
        @change="fetchList"
      />
    </el-card>

    <!-- Create Task Dialog -->
    <el-dialog v-model="showCreateDialog" title="新建推理任务" width="520px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="100px">
        <el-form-item label="选择图像" prop="imageId">
          <el-select v-model="form.imageId" placeholder="选择一张图像" style="width:100%"
            filterable>
            <el-option v-for="img in images" :key="img.id"
              :label="`${img.fileName} (${img.width}x${img.height})`" :value="img.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="流水线类型" prop="pipelineType">
          <el-select v-model="form.pipelineType" style="width:100%">
            <el-option label="全流程推理（检测+场景图+风险）" value="full_pipeline" />
            <el-option label="仅场景图推理" value="scene_graph_only" />
            <el-option label="仅风险推理" value="reasoning_only" />
            <el-option label="仅目标检测" value="detector_only" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreate" :loading="creating">创建任务</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getDetectionTaskList, deleteDetectionTask, createDetectionTask } from '@/api/detection'
import { getImageList } from '@/api/image'

const router = useRouter()
const loading = ref(false)
const list = ref([])
const total = ref(0)
const pageNum = ref(1)
const pageSize = ref(10)
const filterStatus = ref('')

const showCreateDialog = ref(false)
const creating = ref(false)
const formRef = ref(null)
const form = reactive({ imageId: null, pipelineType: 'full_pipeline' })
const rules = {
  imageId: [{ required: true, message: '请选择图像', trigger: 'change' }],
  pipelineType: [{ required: true, message: '请选择流水线类型', trigger: 'change' }],
}
const images = ref([])

function statusTag(s) {
  const map = { pending: 'info', running: 'warning', completed: 'success', failed: 'danger' }
  return map[s] || 'info'
}
function statusLabel(s) {
  const map = { pending: '等待中', running: '运行中', completed: '已完成', failed: '失败' }
  return map[s] || s
}
function pipelineTag(t) {
  const map = { full_pipeline: 'success', scene_graph_only: 'warning', reasoning_only: 'info', detector_only: '' }
  return map[t] || 'info'
}
function pipelineLabel(t) {
  const map = { full_pipeline: '全流程', scene_graph_only: '场景图', reasoning_only: '风险推理', detector_only: '目标检测' }
  return map[t] || t
}

async function fetchList() {
  loading.value = true
  try {
    const params = { pageNum: pageNum.value, pageSize: pageSize.value }
    if (filterStatus.value) params.status = filterStatus.value
    const res = await getDetectionTaskList(params)
    list.value = res.rows
    total.value = res.total
  } finally {
    loading.value = false
  }
}

async function openCreateDialog() {
  const imgRes = await getImageList({ pageSize: 100 })
  images.value = imgRes.rows || []
  form.imageId = images.value[0]?.id || null
  form.pipelineType = 'full_pipeline'
  showCreateDialog.value = true
}

async function handleCreate() {
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return
  creating.value = true
  try {
    await createDetectionTask({
      imageId: form.imageId,
      pipelineType: form.pipelineType,
    })
    ElMessage.success('推理任务已创建')
    showCreateDialog.value = false
    fetchList()
  } finally {
    creating.value = false
  }
}

function viewResults(row) {
  router.push(`/detection/result?taskId=${row.id}`)
}

async function handleDelete(row) {
  await ElMessageBox.confirm('确定删除该任务及其所有相关数据？', '警告', { type: 'warning' })
  await deleteDetectionTask(row.id)
  ElMessage.success('删除成功')
  fetchList()
}

onMounted(fetchList)
</script>
