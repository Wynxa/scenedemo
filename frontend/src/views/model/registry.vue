<template>
  <div class="model-page">
    <el-card shadow="never">
      <div class="toolbar">
        <span class="page-desc">管理 YOLO 目标检测模型，注册后可独立调用推理接口</span>
        <el-button type="primary" @click="openCreate">
          <el-icon><Plus /></el-icon> 注册模型
        </el-button>
      </div>
    </el-card>

    <el-card shadow="never" style="margin-top:12px">
      <el-table :data="list" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column label="模型" min-width="180">
          <template #default="{ row }">
            <span class="model-name">{{ row.name }}</span>
            <span class="model-version">v{{ row.version }}</span>
          </template>
        </el-table-column>
        <el-table-column label="类型" width="120">
          <template #default="{ row }">
            <el-tag size="small" :type="row.serviceType === 'yolo_detection' ? '' : 'info'">
              {{ row.serviceType === 'yolo_detection' ? 'YOLO 检测' : row.serviceType }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="checkpointPath" label="Checkpoint" min-width="180">
          <template #default="{ row }">
            <code class="mono">{{ row.checkpointPath }}</code>
          </template>
        </el-table-column>
        <el-table-column label="检测参数" width="140">
          <template #default="{ row }">
            <span v-if="row.configJson?.confidenceThreshold" class="cfg-text">
              阈值 {{ row.configJson.confidenceThreshold }} | IOU {{ row.configJson.iouThreshold || '-' }}
            </span>
            <span v-else class="cfg-empty">-</span>
          </template>
        </el-table-column>
        <el-table-column label="指标" width="140">
          <template #default="{ row }">
            <span v-if="row.metricsJson?.mAP50" class="metric">
              mAP50 {{ (row.metricsJson.mAP50 * 100).toFixed(1) }}%
            </span>
            <span v-else class="cfg-empty">-</span>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="80">
          <template #default="{ row }">
            <el-tag size="small" :type="row.status === 'active' ? 'success' : 'info'">
              {{ row.status === 'active' ? '活跃' : '归档' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="300" fixed="right">
          <template #default="{ row }">
            <el-button size="small" link type="warning"
              @click="openDetect(row)"
              :disabled="row.serviceType !== 'yolo_detection'">
              <el-icon><Camera /></el-icon> 快速检测
            </el-button>
            <el-button size="small" link type="success"
              @click="handleActivate(row)" :disabled="row.status === 'active'">
              激活
            </el-button>
            <el-button size="small" link type="primary" @click="openEdit(row)">编辑</el-button>
            <el-button size="small" link type="danger" @click="handleDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination style="margin-top:16px;justify-content:flex-end"
        v-model:current-page="pageNum" v-model:page-size="pageSize" :total="total"
        layout="total, prev, pager, next" @change="fetchList" />
    </el-card>

    <!-- Register/Edit Dialog -->
    <el-dialog
      v-model="showDialog"
      :title="editId ? '编辑模型' : '注册 YOLO 检测模型'"
      width="620px" destroy-on-close
    >
      <el-form ref="formRef" :model="form" :rules="rules" label-width="110px" label-position="left">
        <el-row :gutter="16">
          <el-col :span="14">
            <el-form-item label="模型名称" prop="name">
              <el-input v-model="form.name" placeholder="如 Helmet-YOLOv12" />
            </el-form-item>
          </el-col>
          <el-col :span="10">
            <el-form-item label="版本" prop="version">
              <el-input v-model="form.version" placeholder="1.0.0" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="Checkpoint 路径" prop="checkpointPath">
          <el-input v-model="form.checkpointPath" placeholder="checkpoints/helmet_yolo.pt" />
        </el-form-item>

        <el-divider content-position="left">检测参数</el-divider>

        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="置信度阈值">
              <el-slider v-model="cfg.confidenceThreshold" :min="0.05" :max="0.95" :step="0.05" show-input />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="IOU 阈值">
              <el-slider v-model="cfg.iouThreshold" :min="0.1" :max="0.9" :step="0.05" show-input />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="输入尺寸">
          <el-select v-model="cfg.imageSize" style="width:140px">
            <el-option :value="320" label="320" />
            <el-option :value="640" label="640" />
            <el-option :value="1280" label="1280" />
          </el-select>
        </el-form-item>
        <el-form-item label="检测类别">
          <el-select v-model="cfg.classNames" multiple filterable allow-create
            placeholder="输入类别名后回车添加" style="width:100%">
            <el-option label="helmet" value="helmet" />
            <el-option label="person" value="person" />
            <el-option label="vest" value="vest" />
            <el-option label="glove" value="glove" />
          </el-select>
        </el-form-item>

        <el-divider content-position="left">评估指标</el-divider>

        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="mAP50">
              <el-input-number v-model="mtr.mAP50" :min="0" :max="1" :step="0.001" :precision="4" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="mAP50-95">
              <el-input-number v-model="mtr.mAP50_95" :min="0" :max="1" :step="0.001" :precision="4" style="width:100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="Precision">
              <el-input-number v-model="mtr.precision" :min="0" :max="1" :step="0.001" :precision="4" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Recall">
              <el-input-number v-model="mtr.recall" :min="0" :max="1" :step="0.001" :precision="4" style="width:100%" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">取消</el-button>
        <el-button type="primary" @click="handleSave" :loading="saving">
          {{ editId ? '保存' : '注册' }}
        </el-button>
      </template>
    </el-dialog>

    <!-- Quick Detect Dialog -->
    <el-dialog
      v-model="showDetect"
      :title="`快速检测 — ${detectModel?.name || ''}`"
      width="820px" destroy-on-close
    >
      <div class="detect-layout">
        <div class="detect-left">
          <el-upload
            class="detect-uploader"
            :auto-upload="false" :show-file-list="false"
            :on-change="handleDetectFile" accept="image/*" drag
          >
            <img v-if="detectPreview" :src="detectPreview" class="detect-preview" />
            <div v-else class="detect-placeholder">
              <el-icon :size="40"><UploadFilled /></el-icon>
              <p>拖拽或点击上传图片</p>
            </div>
          </el-upload>
          <el-button type="primary" :disabled="!detectFile" :loading="detecting"
            style="width:100%;margin-top:12px" @click="runDetect">
            <el-icon><VideoPlay /></el-icon> 开始检测
          </el-button>
        </div>
        <div class="detect-right">
          <div v-if="!detectResults" class="detect-empty">
            <el-icon :size="36"><PictureFilled /></el-icon>
            <p>上传图片后点击检测</p>
          </div>
          <div v-else-if="!detectResults.length" class="detect-empty">
            <el-icon :size="36"><CircleCheckFilled /></el-icon>
            <p>未检测到目标</p>
          </div>
          <div v-else class="detect-list">
            <div class="detect-summary">
              检测到 <strong>{{ detectResults.length }}</strong> 个目标
            </div>
            <div v-for="(d, i) in detectResults" :key="i" class="detect-item">
              <span class="detect-idx">{{ i + 1 }}</span>
              <el-tag size="small" :type="d.labelName === 'helmet' ? 'success' : d.labelName === 'person' ? 'warning' : 'info'">
                {{ d.labelName }}
              </el-tag>
              <code class="mono bbox">{{ fmtBbox(d.bbox) }}</code>
              <span class="detect-conf">{{ (d.confidence * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Camera, VideoPlay, UploadFilled, PictureFilled, CircleCheckFilled } from '@element-plus/icons-vue'
import { getModelList, registerModel, updateModel, activateModel, deleteModel } from '@/api/model'
import request from '@/api/request'

const loading = ref(false)
const list = ref([])
const total = ref(0)
const pageNum = ref(1)
const pageSize = ref(10)

// Register/Edit form
const showDialog = ref(false)
const editId = ref(0)
const saving = ref(false)
const formRef = ref(null)
const form = reactive({ name: '', version: '1.0.0', checkpointPath: '' })
const cfg = reactive({ confidenceThreshold: 0.5, iouThreshold: 0.45, imageSize: 640, classNames: [] })
const mtr = reactive({ mAP50: null, mAP50_95: null, precision: null, recall: null })
const rules = {
  name: [{ required: true, message: '请输入模型名称', trigger: 'blur' }],
  checkpointPath: [{ required: true, message: '请输入 checkpoint 路径', trigger: 'blur' }],
  version: [{ required: true, message: '请输入版本号', trigger: 'blur' }],
}

// Quick Detect
const showDetect = ref(false)
const detectModel = ref(null)
const detectFile = ref(null)
const detectPreview = ref('')
const detectResults = ref(null)
const detecting = ref(false)

async function fetchList() {
  loading.value = true
  try {
    const res = await getModelList({ pageNum: pageNum.value, pageSize: pageSize.value })
    list.value = res.rows; total.value = res.total
  } finally { loading.value = false }
}

function resetForm() {
  editId.value = 0
  form.name = ''; form.version = '1.0.0'; form.checkpointPath = ''
  cfg.confidenceThreshold = 0.5; cfg.iouThreshold = 0.45; cfg.imageSize = 640; cfg.classNames = []
  mtr.mAP50 = null; mtr.mAP50_95 = null; mtr.precision = null; mtr.recall = null
}

function openCreate() { resetForm(); showDialog.value = true }

function openEdit(row) {
  editId.value = row.id
  form.name = row.name; form.version = row.version; form.checkpointPath = row.checkpointPath
  const c = row.configJson || {}
  cfg.confidenceThreshold = c.confidenceThreshold ?? 0.5
  cfg.iouThreshold = c.iouThreshold ?? 0.45
  cfg.imageSize = c.imageSize ?? 640
  cfg.classNames = c.classNames || []
  const m = row.metricsJson || {}
  mtr.mAP50 = m.mAP50 ?? null; mtr.mAP50_95 = m.mAP50_95 ?? null
  mtr.precision = m.precision ?? null; mtr.recall = m.recall ?? null
  showDialog.value = true
}

async function handleSave() {
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return
  saving.value = true
  try {
    const payload = {
      name: form.name, version: form.version, checkpointPath: form.checkpointPath,
      serviceType: 'yolo_detection', runtimeType: 'local_python',
      configJson: { confidenceThreshold: cfg.confidenceThreshold, iouThreshold: cfg.iouThreshold, imageSize: cfg.imageSize, classNames: cfg.classNames },
      metricsJson: { mAP50: mtr.mAP50, mAP50_95: mtr.mAP50_95, precision: mtr.precision, recall: mtr.recall },
    }
    if (editId.value) {
      await updateModel(editId.value, payload)
      ElMessage.success('模型已更新')
    } else {
      await registerModel(payload)
      ElMessage.success('模型注册成功')
    }
    showDialog.value = false
    fetchList()
  } finally { saving.value = false }
}

async function handleActivate(row) {
  await activateModel(row.id)
  ElMessage.success(`"${row.name}" 已激活`)
  fetchList()
}

async function handleDelete(row) {
  await ElMessageBox.confirm(`确定删除 "${row.name}"？`, '警告', { type: 'warning' })
  await deleteModel(row.id)
  ElMessage.success('已删除')
  fetchList()
}

// Quick Detect
function openDetect(row) {
  detectModel.value = row
  detectFile.value = null
  detectPreview.value = ''
  detectResults.value = null
  showDetect.value = true
}

function handleDetectFile(file) {
  detectFile.value = file.raw
  detectPreview.value = URL.createObjectURL(file.raw)
  detectResults.value = null
}

async function runDetect() {
  if (!detectFile.value || !detectModel.value) return
  detecting.value = true
  try {
    const fd = new FormData()
    fd.append('file', detectFile.value)
    const res = await request.post(`/model/${detectModel.value.id}/detect`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    detectResults.value = res.detections || []
    if (detectResults.value.length) {
      ElMessage.success(`检测完成，${detectResults.value.length} 个目标`)
    } else {
      ElMessage.info('未检测到目标')
    }
  } finally { detecting.value = false }
}

function fmtBbox(b) { return b ? b.map(v => Math.round(v)).join(', ') : '-' }

onMounted(fetchList)
</script>

<style scoped>
.toolbar { display: flex; justify-content: space-between; align-items: center; }
.page-desc { font-size: 13px; color: #94a3b8; }
.model-name { font-weight: 600; color: #1e293b; }
.model-version { font-size: 11px; color: #94a3b8; margin-left: 6px; font-family: 'JetBrains Mono', monospace; }
.mono { font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; background: #f1f5f9; padding: 2px 6px; border-radius: 3px; color: #334155; }
.cfg-text { font-size: 12px; color: #475569; }
.cfg-empty { font-size: 12px; color: #cbd5e1; }
.metric { font-size: 12px; color: #22c55e; font-weight: 500; font-family: 'JetBrains Mono', monospace; }

.detect-layout { display: grid; grid-template-columns: 360px 1fr; gap: 20px; }
.detect-preview { max-width: 100%; max-height: 280px; object-fit: contain; border-radius: 4px; }
.detect-placeholder { display: flex; flex-direction: column; align-items: center; gap: 10px; color: #94a3b8; padding: 30px 0; font-size: 13px; }
.detect-right { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; overflow-y: auto; max-height: 400px; min-height: 200px; }
.detect-empty { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 50px 0; color: #94a3b8; font-size: 13px; }
.detect-summary { font-size: 13px; color: #475569; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #f1f5f9; }
.detect-summary strong { color: #ff6b35; font-size: 16px; font-family: 'JetBrains Mono', monospace; }
.detect-item { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 13px; }
.detect-idx { width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; background: #f1f5f9; border-radius: 50%; font-size: 11px; font-weight: 600; color: #64748b; font-family: 'JetBrains Mono', monospace; }
.detect-conf { font-size: 12px; font-weight: 600; color: #22c55e; font-family: 'JetBrains Mono', monospace; margin-left: auto; }
.bbox { font-size: 11px; }
</style>
