<template>
  <section class="command-center">
    <header class="page-intro">
      <div>
        <p class="eyebrow">SAFE SCENE / CONTROL ROOM</p>
        <h1>施工现场智能安全工作台</h1>
        <p class="intro-copy">集中监控视觉检测、场景图构建与风险推理任务。</p>
      </div>
      <div class="intro-actions">
        <div class="system-state"><i></i><span>服务在线</span><b>自动刷新 5s</b></div>
        <el-button type="primary" @click="$router.push('/inference/pipeline')">
          <el-icon><VideoPlay /></el-icon> 新建推理任务
        </el-button>
      </div>
    </header>

    <section class="overview-grid">
      <article class="metric-card accent-blue">
        <span class="metric-label">已接入现场图像</span><strong>{{ stats.totalImages || 0 }}</strong>
        <small>累计识别实体 {{ stats.totalObjects || 0 }} 个</small>
      </article>
      <article class="metric-card accent-teal">
        <span class="metric-label">场景语义关系</span><strong>{{ stats.totalRelations || 0 }}</strong>
        <small>模型服务 {{ stats.activeModels || 0 }} 个在线</small>
      </article>
      <article class="metric-card accent-amber">
        <span class="metric-label">推理任务总量</span><strong>{{ stats.totalInferenceTasks || 0 }}</strong>
        <small><em class="running-dot"></em>{{ stats.runningTasks || 0 }} 个任务正在执行</small>
      </article>
      <article class="metric-card accent-red">
        <span class="metric-label">风险现场</span><strong>{{ stats.unsafeImages || 0 }}</strong>
        <small>已完成评估 {{ stats.totalSummaries || 0 }} 张图像</small>
      </article>
    </section>

    <section class="workspace-grid">
      <article class="panel live-panel">
        <div class="panel-head">
          <div><p class="panel-kicker">LIVE INFERENCE</p><h2>实时推理态势</h2></div>
          <span class="live-status" :class="{ idle: !activeTask }"><i></i>{{ activeTask ? '实时跟踪中' : '等待任务' }}</span>
        </div>

        <template v-if="activeTask">
          <div class="task-banner">
            <div class="task-ident"><span>任务</span><code>#{{ activeTask.id }}</code><b>{{ pipelineLabel(activeTask.pipelineType) }}</b></div>
            <div class="task-state" :class="activeTask.status">{{ statusLabel(activeTask.status) }}</div>
          </div>
          <div class="stage-track" aria-label="推理阶段进度">
            <div v-for="(stage, index) in stages" :key="stage.key" class="stage-item" :class="stageClass(stage, index)">
              <div class="stage-marker"><el-icon v-if="stageState(stage, index) === 'completed'"><CircleCheckFilled /></el-icon><span v-else>{{ index + 1 }}</span></div>
              <div><b>{{ stage.label }}</b><small>{{ stage.desc }}</small></div>
            </div>
          </div>
          <div class="stream-row">
            <div class="stream-visual"><span class="scan-line"></span><el-icon><DataAnalysis /></el-icon><b>推理数据流</b><small>{{ stageLabel(activeTask.currentStage) }}</small></div>
            <div class="stream-output">
              <span>当前输出</span>
              <strong>{{ activeSummary ? (activeSummary.hasUnsafeBehavior ? '发现风险行为' : '未发现风险行为') : stageOutput }}</strong>
              <small v-if="activeSummary">风险人员 {{ activeSummary.unsafeWorkerCount || 0 }} · {{ activeSummary.highestRiskCategory || '—' }}</small>
              <small v-else>任务进度将随服务端阶段状态更新</small>
            </div>
          </div>
          <div class="task-footer"><span>创建于 {{ activeTask.createTime || '—' }}</span><el-button text type="primary" @click="$router.push('/inference/pipeline')">打开推理工作区 <el-icon><ArrowRight /></el-icon></el-button></div>
        </template>
        <div v-else class="empty-live"><el-icon><VideoPlay /></el-icon><strong>当前没有运行中的推理任务</strong><span>创建任务后，这里会实时显示检测、场景图和风险推理阶段。</span><el-button plain @click="$router.push('/inference/pipeline')">发起推理</el-button></div>
      </article>

      <aside class="panel alert-panel">
        <div class="panel-head"><div><p class="panel-kicker">RISK WATCH</p><h2>风险告警概览</h2></div><el-icon class="warning-icon"><WarningFilled /></el-icon></div>
        <div class="risk-score"><span>风险图像占比</span><strong>{{ riskRate }}<small>%</small></strong><div class="risk-bar"><i :style="{ width: `${riskRate}%` }"></i></div></div>
        <div class="alert-list">
          <div><span class="alert-chip critical"></span><p><b>高风险场景</b><small>{{ stats.unsafeImages || 0 }} 个已识别现场</small></p><strong>{{ stats.unsafeImages || 0 }}</strong></div>
          <div><span class="alert-chip normal"></span><p><b>安全完成任务</b><small>风险核验已闭环</small></p><strong>{{ stats.safeImages || 0 }}</strong></div>
          <div><span class="alert-chip queued"></span><p><b>待处理任务</b><small>等待进入推理队列</small></p><strong>{{ pendingCount }}</strong></div>
        </div>
      </aside>
    </section>

    <section class="bottom-grid">
      <article class="panel chart-panel"><div class="panel-head"><div><p class="panel-kicker">VISION ASSETS</p><h2>识别实体分布</h2></div></div><div ref="entityChartRef" class="chart-box"></div></article>
      <article class="panel queue-panel">
        <div class="panel-head"><div><p class="panel-kicker">TASK QUEUE</p><h2>最近推理任务</h2></div><el-button text type="primary" @click="$router.push('/inference/pipeline')">查看工作区</el-button></div>
        <div class="task-table">
          <div v-for="task in recentTasks.slice(0, 5)" :key="task.id" class="task-row" :class="{ selected: activeTask?.id === task.id }">
            <div class="task-number">{{ String(task.id).padStart(2, '0') }}</div><div class="task-meta"><b>{{ pipelineLabel(task.pipelineType) }}</b><small>{{ task.createTime || '—' }} · {{ stageLabel(task.currentStage) }}</small></div><span class="status-pill" :class="task.status"><i></i>{{ statusLabel(task.status) }}</span>
          </div>
          <el-empty v-if="!recentTasks.length" description="暂无推理任务" :image-size="70" />
        </div>
      </article>
    </section>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ArrowRight, CircleCheckFilled, DataAnalysis, VideoPlay, WarningFilled } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import { getDashboardStats, getEntityDistribution } from '@/api/dashboard'
import { getInferenceTask } from '@/api/infer'

const entityChartRef = ref(null)
const stats = ref({})
const recentTasks = ref([])
const activeDetail = ref(null)
let entityChart
let refreshTimer

const stages = [
  { key: 'detector', label: '目标检测', desc: '实体与安全设施识别' },
  { key: 'scene_graph', label: '场景图构建', desc: '关系抽取与结构化表达' },
  { key: 'reasoning', label: '危险推理', desc: '规则匹配与风险判定' },
]
const activeTask = computed(() => recentTasks.value.find(task => task.status === 'running') || recentTasks.value.find(task => task.status === 'pending') || null)
const activeSummary = computed(() => activeDetail.value?.summary || null)
const pendingCount = computed(() => recentTasks.value.filter(task => task.status === 'pending').length)
const riskRate = computed(() => { const total = Number(stats.value.totalSummaries || 0); return total ? Math.round((Number(stats.value.unsafeImages || 0) / total) * 100) : 0 })
const stageOutput = computed(() => ({ detector: '正在提取现场实体', scene_graph: '正在构建语义关系', reasoning: '正在进行风险判定' })[activeTask.value?.currentStage] || '任务等待调度')

const pipelineLabel = type => ({ full_pipeline: '全流程推理', scene_graph_only: '场景图推理', reasoning_only: '风险推理' })[type] || type || '推理任务'
const stageLabel = stage => ({ detector: '目标检测', scene_graph: '场景图构建', reasoning: '危险推理' })[stage] || stage || '待调度'
const statusLabel = status => ({ pending: '等待中', running: '运行中', completed: '已完成', failed: '失败' })[status] || status
function stageState(stage, index) {
  const row = activeDetail.value?.stageRuns?.find(item => item.stageName === stage.key)
  if (row?.status === 'completed') return 'completed'
  if (row?.status === 'failed') return 'failed'
  if (activeTask.value?.currentStage === stage.key) return 'active'
  const currentIndex = stages.findIndex(item => item.key === activeTask.value?.currentStage)
  return currentIndex > index ? 'completed' : 'pending'
}
function stageClass(stage, index) { return `is-${stageState(stage, index)}` }

async function refreshDashboard() {
  try {
    const data = await getDashboardStats()
    stats.value = data || {}
    recentTasks.value = data?.recentTasks || []
    if (activeTask.value) activeDetail.value = await getInferenceTask(activeTask.value.id)
    else activeDetail.value = null
  } catch { /* keep last good operating view */ }
}
async function renderEntityChart() {
  try {
    const data = await getEntityDistribution()
    entityChart = echarts.init(entityChartRef.value)
    entityChart.setOption({
      grid: { left: 32, right: 14, top: 12, bottom: 26 }, tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: data.categories.map(item => item.replace('_', ' ')), axisLine: { lineStyle: { color: '#d6dde5' } }, axisLabel: { color: '#657387', fontSize: 10 } },
      yAxis: { type: 'value', splitLine: { lineStyle: { color: '#edf1f4' } }, axisLabel: { color: '#8793a3', fontSize: 10 } },
      series: [{ type: 'bar', data: data.values, barWidth: '45%', itemStyle: { color: '#237a8b', borderRadius: [3, 3, 0, 0] }, emphasis: { itemStyle: { color: '#e5962d' } } }],
    })
  } catch { /* unavailable data should not block the workspace */ }
}
function onResize() { entityChart?.resize() }
onMounted(async () => { await Promise.all([refreshDashboard(), renderEntityChart()]); refreshTimer = window.setInterval(refreshDashboard, 5000); window.addEventListener('resize', onResize) })
onBeforeUnmount(() => { window.clearInterval(refreshTimer); window.removeEventListener('resize', onResize); entityChart?.dispose() })
</script>

<style scoped>
.command-center{--ink:#18222d;--muted:#718095;--line:#dce3e9;--panel:#fff;--blue:#277b8b;--amber:#e79a30;--red:#c74c45;display:flex;flex-direction:column;gap:16px;color:var(--ink)}
.page-intro,.panel-head,.intro-actions,.system-state,.task-banner,.task-footer,.task-row{display:flex;align-items:center}.page-intro{justify-content:space-between;padding:4px 2px 2px}.eyebrow,.panel-kicker{margin:0 0 5px;color:#278092;font:700 10px/1.2 var(--font-mono);letter-spacing:1.15px}.page-intro h1,.panel h2{margin:0;font-weight:700}.page-intro h1{font-size:23px;letter-spacing:.02em}.intro-copy{margin:6px 0 0;color:var(--muted);font-size:13px}.intro-actions{gap:14px}.system-state{gap:7px;color:#526174;font-size:12px}.system-state i,.live-status i,.running-dot,.status-pill i{width:7px;height:7px;border-radius:50%;background:#32a46b;display:inline-block;box-shadow:0 0 0 4px rgba(50,164,107,.11)}.system-state b{margin-left:5px;padding-left:11px;border-left:1px solid var(--line);font-weight:500;color:#8995a4}.overview-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric-card{position:relative;min-height:118px;padding:17px 18px;background:var(--panel);border:1px solid var(--line);border-radius:5px;overflow:hidden}.metric-card:before{content:"";position:absolute;top:0;left:0;width:4px;height:100%;background:var(--accent)}.accent-blue{--accent:#287f91}.accent-teal{--accent:#42a693}.accent-amber{--accent:#e6a136}.accent-red{--accent:#c74c45}.metric-label{color:#667589;font-size:12px}.metric-card strong{display:block;margin:6px 0 4px;font:700 30px/1 var(--font-mono)}.metric-card small{color:#8592a1;font-size:11px}.running-dot{margin-right:5px;background:#e6a136;box-shadow:0 0 0 4px rgba(230,161,54,.12)}
.workspace-grid,.bottom-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(300px,.8fr);gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:5px;box-shadow:0 1px 2px rgba(14,28,44,.025)}.panel-head{justify-content:space-between;padding:16px 18px 13px;border-bottom:1px solid #e9eef1}.panel h2{font-size:15px}.warning-icon{font-size:21px;color:var(--amber)}.live-status{display:flex;align-items:center;gap:6px;color:#21836f;font-size:11px;font-weight:600}.live-status.idle{color:#8a97a5}.live-status.idle i{background:#aab4be;box-shadow:none}.task-banner{justify-content:space-between;padding:12px 18px;background:#f7fafb;border-bottom:1px solid #e8eef1}.task-ident{display:flex;align-items:center;gap:8px;font-size:11px;color:#7a8898}.task-ident code{font-weight:700;color:#334254}.task-ident b{font-size:11px;color:#3b5662}.task-state,.status-pill{font-size:11px;font-weight:700}.task-state.running{color:#c27b12}.task-state.pending{color:#5c7188}.task-state.completed{color:#258358}.task-state.failed{color:var(--red)}.stage-track{display:grid;grid-template-columns:repeat(3,1fr);padding:23px 18px 17px}.stage-item{position:relative;display:flex;gap:10px;min-width:0}.stage-item:not(:last-child):after{content:"";position:absolute;top:13px;left:30px;width:calc(100% - 28px);height:1px;background:#d9e1e6}.stage-marker{z-index:1;display:grid;place-items:center;width:27px;height:27px;flex:0 0 27px;border-radius:50%;border:1px solid #cbd5dc;background:#fff;color:#8b98a5;font:700 11px var(--font-mono)}.stage-item b{display:block;font-size:12px}.stage-item small{display:block;margin-top:3px;padding-right:7px;color:#8490a0;font-size:10px;line-height:1.4}.stage-item.is-active .stage-marker{border-color:#e69a30;background:#e69a30;color:#fff;box-shadow:0 0 0 4px rgba(230,154,48,.13)}.stage-item.is-active b{color:#b66f0d}.stage-item.is-completed .stage-marker{border-color:#2b8c78;background:#2b8c78;color:#fff}.stage-item.is-completed:after{background:#2b8c78}.stage-item.is-failed .stage-marker{border-color:var(--red);color:var(--red)}.stream-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 18px 12px}.stream-visual{position:relative;min-height:91px;display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden;background:#172b36;color:#a7d7dd}.stream-visual:before{content:"";position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent 0 21px,rgba(129,210,214,.07) 22px 23px),repeating-linear-gradient(0deg,transparent 0 21px,rgba(129,210,214,.07) 22px 23px)}.scan-line{position:absolute;width:100%;height:2px;background:#54d1c4;box-shadow:0 0 12px #54d1c4;animation:scan 2.4s linear infinite}.stream-visual :not(.scan-line){position:relative}.stream-visual .el-icon{font-size:21px}.stream-visual b{font-size:11px;margin-top:3px}.stream-visual small{font:10px var(--font-mono);color:#76aeb7}.stream-output{padding:14px;background:#f4f7f8;border-left:3px solid #2b8898}.stream-output span,.stream-output small{display:block;color:#788698;font-size:10px}.stream-output strong{display:block;margin:7px 0;font-size:14px}.task-footer{justify-content:space-between;padding:0 18px 12px;color:#8491a0;font:10px var(--font-mono)}.empty-live{display:flex;min-height:209px;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:#788798;text-align:center;font-size:12px}.empty-live .el-icon{font-size:28px;color:#3d8d99}.empty-live strong{color:#3b4b5e;font-size:13px}.empty-live .el-button{margin-top:5px}.risk-score{padding:17px 18px;border-bottom:1px solid #e9eef1}.risk-score span{display:block;color:#718095;font-size:11px}.risk-score strong{display:block;margin:3px 0 10px;color:#c14c45;font:700 28px/1 var(--font-mono)}.risk-score strong small{font-size:13px}.risk-bar{height:5px;background:#edf0f2}.risk-bar i{display:block;height:100%;background:#c74c45}.alert-list>div{display:flex;align-items:center;gap:10px;padding:13px 18px;border-bottom:1px solid #edf1f3}.alert-list>div:last-child{border:none}.alert-chip{width:8px;height:28px;border-radius:1px}.critical{background:#c74c45}.normal{background:#3a9a77}.queued{background:#d79a34}.alert-list p{flex:1;margin:0}.alert-list b,.alert-list small{display:block}.alert-list b{font-size:12px}.alert-list small{margin-top:2px;color:#8995a4;font-size:10px}.alert-list strong{font:700 17px var(--font-mono)}.chart-box{height:212px}.queue-panel .panel-head{padding-bottom:12px}.task-table{padding:1px 18px 8px}.task-row{gap:11px;padding:11px 0;border-bottom:1px solid #edf1f3}.task-row:last-child{border:none}.task-row.selected{background:linear-gradient(90deg,rgba(39,123,139,.06),transparent);margin:0 -8px;padding-left:8px;padding-right:8px}.task-number{display:grid;place-items:center;width:28px;height:28px;background:#f0f4f5;color:#57818a;font:700 10px var(--font-mono)}.task-meta{flex:1;min-width:0}.task-meta b,.task-meta small{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.task-meta b{font-size:11px}.task-meta small{margin-top:3px;color:#8995a3;font-size:10px}.status-pill{display:flex;align-items:center;gap:5px;color:#6f7d8d}.status-pill i{width:5px;height:5px;box-shadow:none;background:#aab4be}.status-pill.running{color:#b87009}.status-pill.running i{background:#e6a136}.status-pill.completed{color:#28835b}.status-pill.completed i{background:#2d9a6e}.status-pill.failed{color:#c74c45}.status-pill.failed i{background:#c74c45}@keyframes scan{0%{transform:translateY(-46px)}100%{transform:translateY(46px)}}@media(max-width:1100px){.overview-grid{grid-template-columns:repeat(2,1fr)}.workspace-grid,.bottom-grid{grid-template-columns:1fr}.alert-panel{display:none}}@media(max-width:680px){.page-intro{align-items:flex-start;gap:14px;flex-direction:column}.overview-grid{grid-template-columns:1fr}.intro-actions{width:100%;justify-content:space-between}.stage-track{grid-template-columns:1fr;gap:14px}.stage-item:after{display:none}.stream-row{grid-template-columns:1fr}.system-state b{display:none}}
</style>
