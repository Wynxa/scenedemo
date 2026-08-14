<template>
  <div class="pipeline-page">
    <!-- Top: Image Selection & Pipeline Control -->
    <div class="pipeline-top">
      <el-card shadow="never" class="image-card">
        <template #header>
          <span class="card-title">输入图像</span>
        </template>
        <div class="image-input-area">
          <el-upload
            class="image-uploader"
            :auto-upload="false"
            :show-file-list="false"
            :on-change="handleFileSelect"
            accept="image/*"
            drag
          >
            <div v-if="!previewUrl" class="upload-placeholder">
              <el-icon :size="40"><UploadFilled /></el-icon>
              <p>拖拽或点击上传施工现场图像</p>
            </div>
            <img v-else :src="previewUrl" class="preview-image" />
          </el-upload>
          <div v-if="selectedFile" class="image-info">
            <span>{{ selectedFile.name }}</span>
            <span class="image-dims">{{ imgWidth }} x {{ imgHeight }} px</span>
          </div>
        </div>
      </el-card>

      <el-card shadow="never" class="pipeline-card">
        <template #header>
          <div class="card-header-row">
            <span class="card-title">推理流水线</span>
            <el-button
              type="primary"
              :disabled="!selectedFile || running"
              :loading="running"
              @click="runPipeline"
            >
              <el-icon><VideoPlay /></el-icon>
              {{ running ? '运行中...' : '运行全流程' }}
            </el-button>
          </div>
        </template>

        <div class="pipeline-stages">
          <div
            v-for="(stage, idx) in stages"
            :key="stage.key"
            class="stage-node"
            :class="{
              'stage-active': currentStageIdx === idx,
              'stage-done': currentStageIdx > idx,
              'stage-error': currentStageIdx === idx && stageError,
            }"
          >
            <div class="stage-icon">
              <el-icon v-if="currentStageIdx > idx" :size="22"><CircleCheckFilled /></el-icon>
              <el-icon v-else-if="currentStageIdx === idx && stageError" :size="22"><CircleCloseFilled /></el-icon>
              <span v-else class="stage-num">{{ idx + 1 }}</span>
            </div>
            <div class="stage-body">
              <div class="stage-name">{{ stage.label }}</div>
              <div class="stage-desc">{{ stage.desc }}</div>
              <el-progress
                v-if="currentStageIdx === idx && running"
                :percentage="100"
                :indeterminate="true"
                :stroke-width="3"
                style="width: 100px"
              />
              <span v-if="currentStageIdx > idx && stageMetrics[idx]" class="stage-stat">
                {{ stageMetrics[idx] }}
              </span>
            </div>
          </div>
        </div>

        <div v-if="taskResult" class="pipeline-summary">
          <div class="summary-stat">
            <span class="stat-value" :class="{ unsafe: taskResult.summary?.hasUnsafeBehavior }">
              {{ taskResult.summary?.hasUnsafeBehavior ? '发现不安全行为' : '未发现不安全行为' }}
            </span>
            <span class="stat-label">图像级结论</span>
          </div>
          <div class="summary-stat">
            <span class="stat-value">{{ taskResult.summary?.unsafeWorkerCount ?? '-' }}</span>
            <span class="stat-label">不安全工人数</span>
          </div>
          <div class="summary-stat">
            <span class="stat-value risk">{{ taskResult.summary?.highestRiskCategory || '-' }}</span>
            <span class="stat-label">最高风险类别</span>
          </div>
        </div>
      </el-card>
    </div>

    <!-- Bottom: Results -->
    <el-card v-if="taskId" shadow="never" class="results-card">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="场景图" name="sceneGraph">
          <SceneGraphView
            :objects="sceneGraphData.objects"
            :relations="sceneGraphData.relations"
            :hazards="hazardData.workerResults"
            :image-width="imgWidth"
            :image-height="imgHeight"
            :image-url="previewUrl"
          />
        </el-tab-pane>
        <el-tab-pane label="危险推理结果" name="hazards">
          <HazardResultsPanel :worker-results="hazardData.workerResults" />
        </el-tab-pane>
        <el-tab-pane label="检测对象" name="objects">
          <el-table :data="sceneGraphData.objects" stripe size="small">
            <el-table-column prop="objectIndex" label="#" width="50" />
            <el-table-column prop="labelName" label="类别" width="120">
              <template #default="{ row }">
                <el-tag size="small" :type="objectTagType(row.labelName)">{{ row.labelName }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="边界框">
              <template #default="{ row }">
                <code class="data-mono">{{ row.bbox.map(v => v.toFixed(0)).join(', ') }}</code>
              </template>
            </el-table-column>
            <el-table-column prop="score" label="置信度" width="100">
              <template #default="{ row }">
                {{ (row.score * 100).toFixed(1) }}%
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
        <el-tab-pane label="原始数据" name="raw">
          <el-input
            type="textarea"
            :rows="16"
            :model-value="JSON.stringify({ objects: sceneGraphData.objects, relations: sceneGraphData.relations, hazards: hazardData }, null, 2)"
            readonly
          />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { onBeforeUnmount, ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, VideoPlay, CircleCheckFilled, CircleCloseFilled } from '@element-plus/icons-vue'
import { runPipelineSync, getInferenceTask, getTaskSceneGraph, getTaskHazards } from '@/api/infer'
import { uploadImage } from '@/api/image'
import SceneGraphView from './components/SceneGraphView.vue'
import HazardResultsPanel from './components/HazardResultsPanel.vue'

const stages = [
  { key: 'detector', label: '目标检测', desc: '识别图像中的工人、机械、材料及安全设施' },
  { key: 'scene_graph', label: '场景图构建', desc: '推理对象间的空间与功能关系' },
  { key: 'reasoning', label: '危险推理', desc: '基于场景图规则判定不安全行为' },
]

const selectedFile = ref(null)
const previewUrl = ref('')
const imgWidth = ref(0)
const imgHeight = ref(0)
const running = ref(false)
const currentStageIdx = ref(-1)
const stageError = ref(false)
const stageMetrics = ref([])
const taskId = ref(null)
const taskResult = ref(null)
const activeTab = ref('sceneGraph')
const sceneGraphData = reactive({ objects: [], relations: [] })
const hazardData = reactive({ workerResults: [], imageLevelResult: null })
let taskPollTimer = null

function handleFileSelect(file) {
  selectedFile.value = file.raw
  previewUrl.value = URL.createObjectURL(file.raw)
  const img = new Image()
  img.onload = () => {
    imgWidth.value = img.naturalWidth
    imgHeight.value = img.naturalHeight
  }
  img.src = previewUrl.value
  taskId.value = null
  taskResult.value = null
  sceneGraphData.objects = []
  sceneGraphData.relations = []
  hazardData.workerResults = []
}

async function runPipeline() {
  if (!selectedFile.value) return
  running.value = true
  stageError.value = false
  currentStageIdx.value = 0
  taskResult.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('pipelineType', 'full_pipeline')

    // Upload image first
    const imageResult = await uploadImage(formData)
    const imageId = imageResult.id

    // Run pipeline synchronously
    const res = await runPipelineSync({ imageId, runNow: true })
    const task = res.task
    taskId.value = task.id

    if (!res.result) {
      taskPollTimer = window.setInterval(() => pollTask(task.id), 1500)
      await pollTask(task.id)
      return
    }
    currentStageIdx.value = 3

    if (res.result) {
      taskResult.value = {
        detector: res.result.detector || {},
        sceneGraph: res.result.sceneGraph || {},
        reasoning: res.result.reasoning || {},
        summary: res.result.reasoning?.image_level_result || null,
      }
      stageMetrics.value = [
        `${taskResult.value.detector.objects?.length ?? '-'} 个对象`,
        `${taskResult.value.sceneGraph.relationships?.length ?? '-'} 条关系`,
        `${taskResult.value.reasoning.worker_results?.length ?? '-'} 个工人分析`,
      ]
    }

    // Fetch full data
    await Promise.all([
      loadSceneGraph(task.id),
      loadHazards(task.id),
    ])
  } catch (e) {
    stageError.value = true
    ElMessage.error('流水线执行失败: ' + (e.message || '未知错误'))
  } finally {
    if (!taskPollTimer) running.value = false
  }
}

async function pollTask(id) {
  try {
    const data = await getInferenceTask(id)
    const task = data.task
    const index = stages.findIndex(stage => stage.key === task.currentStage)
    currentStageIdx.value = task.status === 'completed' ? stages.length : Math.max(index, 0)

    if (task.status === 'completed') {
      await Promise.all([loadSceneGraph(id), loadHazards(id)])
      running.value = false
      window.clearInterval(taskPollTimer)
      taskPollTimer = null
    } else if (task.status === 'failed') {
      stageError.value = true
      running.value = false
      window.clearInterval(taskPollTimer)
      taskPollTimer = null
      ElMessage.error(task.errorMessage || '推理任务失败')
    }
  } catch {
    // A transient polling error must not cancel the server-side task.
  }
}

onBeforeUnmount(() => {
  if (taskPollTimer) window.clearInterval(taskPollTimer)
})

async function loadSceneGraph(taskId_) {
  try {
    const data = await getTaskSceneGraph(taskId_)
    sceneGraphData.objects = data.objects || []
    sceneGraphData.relations = data.relationships || []
  } catch { /* ignore */ }
}

async function loadHazards(taskId_) {
  try {
    const data = await getTaskHazards(taskId_)
    hazardData.workerResults = data.workerResults || []
    hazardData.imageLevelResult = data.imageLevelResult
    taskResult.value = taskResult.value || {}
    taskResult.value.summary = data.imageLevelResult
  } catch { /* ignore */ }
}

function objectTagType(label) {
  if (!label) return 'info'
  if (/worker|person|工人|人员/i.test(label)) return 'warning'
  if (/helmet|vest|harness|安全帽|安全带|反光衣/i.test(label)) return 'success'
  if (/scaffold|guardrail|edge|ladder|脚手架|护栏|边缘/i.test(label)) return 'danger'
  if (/crane|excavator|machinery|吊车|挖掘机|机械/i.test(label)) return ''
  return 'info'
}
</script>

<style scoped>
.pipeline-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.pipeline-top {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 16px;
}

.card-title {
  font-weight: 600;
  font-size: 15px;
  color: #1e293b;
}

.card-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.image-card {
  min-height: 360px;
}

.image-input-area {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.image-uploader {
  width: 100%;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: #94a3b8;
  padding: 40px 0;
}

.upload-placeholder p {
  font-size: 13px;
  margin: 0;
}

.preview-image {
  max-width: 100%;
  max-height: 300px;
  object-fit: contain;
  border-radius: 4px;
}

.image-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  margin-top: 12px;
  font-size: 13px;
  color: #475569;
}

.image-dims {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
  font-size: 12px;
  color: #94a3b8;
}

/* Pipeline Stages */
.pipeline-stages {
  display: flex;
  flex-direction: column;
  gap: 0;
  padding: 8px 0;
}

.stage-node {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 16px 12px;
  position: relative;
  border-left: 2px solid #e2e8f0;
  margin-left: 19px;
  transition: border-color 0.3s, background 0.3s;
}

.stage-node.stage-active {
  border-left-color: #ff6b35;
  background: rgba(255, 107, 53, 0.04);
}

.stage-node.stage-done {
  border-left-color: #22c55e;
}

.stage-node.stage-error {
  border-left-color: #ef4444;
  background: rgba(239, 68, 68, 0.04);
}

.stage-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-left: -40px;
  background: #f8fafc;
  border-radius: 50%;
}

.stage-num {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #e2e8f0;
  color: #64748b;
  font-size: 13px;
  font-weight: 600;
}

.stage-active .stage-num {
  background: #ff6b35;
  color: #fff;
}

.stage-done .stage-icon {
  color: #22c55e;
}

.stage-error .stage-icon {
  color: #ef4444;
}

.stage-body {
  flex: 1;
}

.stage-name {
  font-weight: 600;
  font-size: 14px;
  color: #1e293b;
  margin-bottom: 2px;
}

.stage-desc {
  font-size: 12px;
  color: #94a3b8;
  margin-bottom: 6px;
}

.stage-stat {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
  font-size: 12px;
  color: #22c55e;
  font-weight: 500;
}

/* Pipeline Summary */
.pipeline-summary {
  display: flex;
  gap: 24px;
  padding: 16px;
  margin-top: 12px;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.summary-stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-value {
  font-size: 20px;
  font-weight: 700;
  color: #22c55e;
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
}

.stat-value.unsafe {
  color: #ef4444;
}

.stat-value.risk {
  color: #ff6b35;
  font-size: 14px;
}

.stat-label {
  font-size: 12px;
  color: #94a3b8;
}

.results-card {
  flex: 1;
  min-height: 400px;
}

.data-mono {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
  font-size: 12px;
  background: #f1f5f9;
  padding: 2px 6px;
  border-radius: 3px;
  color: #334155;
}
</style>
