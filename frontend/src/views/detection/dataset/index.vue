<template>
  <div class="dataset-page">
    <!-- Toolbar -->
    <el-card shadow="never" class="toolbar">
      <el-form :inline="true">
        <el-form-item>
          <el-input v-model="searchName" placeholder="数据集名称" clearable @keyup.enter="fetchList" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchList">查询</el-button>
          <el-button type="success" @click="showCreateDialog = true">新建数据集</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Table -->
    <el-card shadow="never" style="margin-top:12px">
      <el-table :data="list" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="name" label="名称" min-width="180" />
        <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
        <el-table-column prop="imageCount" label="图片数量" width="100" />
        <el-table-column prop="createTime" label="创建时间" width="170" />
        <el-table-column label="操作" width="280">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="$router.push(`/detection/dataset/${row.id}`)">
              管理图片
            </el-button>
            <el-button size="small" type="success" link @click="startDetection(row)">
              开始检测
            </el-button>
            <el-button size="small" type="danger" link @click="handleDelete(row)">删除</el-button>
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

    <!-- Create Dialog -->
    <el-dialog v-model="showCreateDialog" title="新建数据集" width="480px">
      <el-form ref="createFormRef" :model="createForm" :rules="createRules" label-width="80px">
        <el-form-item label="名称" prop="name">
          <el-input v-model="createForm.name" placeholder="数据集名称" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input v-model="createForm.description" type="textarea" :rows="3" placeholder="可选描述" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreate">确定</el-button>
      </template>
    </el-dialog>

    <!-- Create Detection Task Dialog -->
    <el-dialog v-model="showTaskDialog" title="创建检测任务" width="500px">
      <el-form label-width="100px">
        <el-form-item label="数据集">
          <el-input :value="selectedDataset?.name" disabled />
        </el-form-item>
        <el-form-item label="选择模型">
          <el-select v-model="taskModelId" placeholder="选择模型" style="width:100%">
            <el-option v-for="m in modelList" :key="m.id" :label="`${m.name} v${m.version}`" :value="m.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="置信度阈值">
          <el-slider v-model="taskThreshold" :min="0" :max="1" :step="0.05" show-input />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showTaskDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreateTask" :loading="taskCreating">开始检测</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getDatasetList, createDataset, deleteDataset } from '@/api/dataset'
import { getModelList } from '@/api/model'
import { createDetectionTask } from '@/api/detection'

const router = useRouter()
const loading = ref(false)
const list = ref([])
const total = ref(0)
const pageNum = ref(1)
const pageSize = ref(10)
const searchName = ref('')

const showCreateDialog = ref(false)
const createFormRef = ref(null)
const createForm = reactive({ name: '', description: '' })
const createRules = { name: [{ required: true, message: '请输入名称', trigger: 'blur' }] }

const showTaskDialog = ref(false)
const selectedDataset = ref(null)
const taskModelId = ref(null)
const taskThreshold = ref(0.5)
const taskCreating = ref(false)
const modelList = ref([])

async function fetchList() {
  loading.value = true
  try {
    const res = await getDatasetList({ pageNum: pageNum.value, pageSize: pageSize.value, name: searchName.value })
    list.value = res.rows
    total.value = res.total
  } finally {
    loading.value = false
  }
}

async function handleCreate() {
  const valid = await createFormRef.value.validate().catch(() => false)
  if (!valid) return
  await createDataset(createForm)
  ElMessage.success('创建成功')
  showCreateDialog.value = false
  createForm.name = ''
  createForm.description = ''
  fetchList()
}

async function handleDelete(row) {
  await ElMessageBox.confirm('确定删除该数据集？图片也将被删除。', '警告', { type: 'warning' })
  await deleteDataset(row.id)
  ElMessage.success('删除成功')
  fetchList()
}

async function startDetection(row) {
  selectedDataset.value = row
  const res = await getModelList({ pageSize: 100 })
  modelList.value = res.rows || []
  taskModelId.value = modelList.value[0]?.id || null
  showTaskDialog.value = true
}

async function handleCreateTask() {
  taskCreating.value = true
  try {
    await createDetectionTask({
      datasetId: selectedDataset.value.id,
      modelId: taskModelId.value,
      config: { threshold: taskThreshold.value },
    })
    ElMessage.success('检测任务已创建，请到检测任务页面查看进度')
    showTaskDialog.value = false
  } finally {
    taskCreating.value = false
  }
}

onMounted(fetchList)
</script>
