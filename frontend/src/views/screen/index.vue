<template>
  <main class="screen" :class="{ 'has-alert': hasUnsafe }">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">◆</span><div><h1>施工安全智慧监控中心</h1><p>现场图像 · 智能识别 · 安全研判</p></div></div>
      <div class="topbar-right">
        <span class="live"><i></i>{{ task?.status === 'running' ? '正在研判' : '系统在线' }}</span>
        <div class="clock"><b>{{ clock.time }}</b><span>{{ clock.date }} {{ clock.week }}</span></div>
        <button @click="router.push('/dashboard')">返回控制台</button>
      </div>
    </header>

    <section class="taskbar">
      <div><span>当前任务</span><strong>{{ task?.taskName || '等待推理任务' }}</strong><code v-if="task">#{{ task.id }}</code></div>
      <div class="task-steps"><span v-for="step in steps" :key="step.key" :class="stepStatus(step.key)"><i></i>{{ step.label }}</span></div>
    </section>

    <section class="four-stage">
      <article class="panel"><PanelTitle no="01" title="现场原图" subtitle="施工现场输入画面" />
        <div class="media-stage"><img v-if="imageUrl" :src="imageUrl" alt="施工现场原图" /><Empty v-else text="等待现场图像" /></div>
        <footer>{{ image?.fileName || '未选择图像' }} <span>{{ image?.width || '—' }} × {{ image?.height || '—' }}</span></footer>
      </article>

      <article class="panel"><PanelTitle no="02" title="目标检测图" subtitle="识别现场人员与安全防护" />
        <div class="media-stage"><svg v-if="imageUrl" class="image-svg" :viewBox="`0 0 ${canvasWidth} ${canvasHeight}`" preserveAspectRatio="xMidYMid meet"><image :href="imageUrl" x="0" y="0" :width="canvasWidth" :height="canvasHeight" preserveAspectRatio="none" /><g v-for="object in objects" :key="object.id"><rect :x="box(object)[0]" :y="box(object)[1]" :width="boxWidth(object)" :height="boxHeight(object)" fill="transparent" :stroke="objectColor(object)" stroke-width="3" /><text :x="box(object)[0] + 4" :y="Math.max(16, box(object)[1] - 6)" :fill="objectColor(object)" font-size="14" font-weight="700" stroke="#07101a" stroke-width="3" paint-order="stroke">{{ object.labelName }} {{ score(object) }}</text></g></svg><Empty v-else text="等待检测结果" /></div>
        <footer><span class="badge blue">已识别 {{ objects.length }} 个目标</span><span>绿色：防护用品 · 橙色：施工人员</span></footer>
      </article>

      <article class="panel"><PanelTitle no="03" title="场景关系图" subtitle="识别对象与作业关系" />
        <div class="media-stage graph-stage" @wheel.prevent="zoomGraph" @dblclick="resetGraphZoom">
          <svg v-if="objects.length" class="graph-svg" :viewBox="`0 0 ${canvasWidth} ${canvasHeight}`" preserveAspectRatio="xMidYMid meet">
            <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#5c7993" /></marker></defs>
            <g :transform="graphTransform">
              <g v-for="rel in relations" :key="rel.id"><line v-if="fromNode(rel) && toNode(rel)" :x1="nodeX(fromNode(rel))" :y1="nodeY(fromNode(rel))" :x2="nodeX(toNode(rel))" :y2="nodeY(toNode(rel))" stroke="#66849e" stroke-width="2" marker-end="url(#arrow)" /><text v-if="fromNode(rel) && toNode(rel)" :x="(nodeX(fromNode(rel)) + nodeX(toNode(rel))) / 2" :y="(nodeY(fromNode(rel)) + nodeY(toNode(rel))) / 2 - 5" fill="#dbeaf4" font-size="12" text-anchor="middle" stroke="#07111a" stroke-width="3" paint-order="stroke">{{ rel.predicateName }}</text></g>
              <g v-for="object in objects" :key="object.id"><circle :cx="nodeX(object)" :cy="nodeY(object)" :r="nodeRadius(object)" :fill="objectColor(object)" :stroke="isHazardObject(object) ? '#ff3d3d' : '#e3f2fa'" :stroke-width="isHazardObject(object) ? 4 : 2" /><text :x="nodeX(object)" :y="nodeY(object) + nodeRadius(object) + 15" fill="#e1edf6" font-size="12" text-anchor="middle" stroke="#07111a" stroke-width="3" paint-order="stroke">{{ object.labelName }}</text></g>
            </g>
          </svg>
          <span v-if="objects.length" class="zoom-tip">WHEEL ZOOM · DOUBLE CLICK TO RESET</span>
          <Empty v-else text="等待场景关系结果" />
        </div>
        <footer><span class="badge teal">{{ relations.length }} 条场景关系</span><span>连线展示人员、设备与防护关系</span></footer>
      </article>

      <article class="panel result-panel"><PanelTitle no="04" title="安全研判结果" subtitle="面向现场处置的结论" />
        <div class="verdict" :class="hasUnsafe ? 'danger' : summary ? 'safe' : 'pending'"><div class="verdict-icon">{{ hasUnsafe ? '!' : summary ? '✓' : '…' }}</div><p>{{ hasUnsafe ? '发现安全风险' : summary ? '现场作业符合要求' : task?.status === 'running' ? '正在生成研判结果' : '等待研判结果' }}</p><strong>{{ riskTitle }}</strong></div>
        <div class="result-details"><div><span>风险人员</span><b :class="{ red: hasUnsafe }">{{ summary?.unsafeWorkerCount ?? '—' }} 人</b></div><div><span>风险等级</span><b :class="{ red: hasUnsafe }">{{ riskLevel }}</b></div></div>
        <div class="recommendation"><span>现场处置建议</span><p>{{ recommendation }}</p></div>
      </article>
    </section>

    <section class="bottomline"><div class="result-summary"><i :class="{ red: hasUnsafe }"></i><strong>{{ hasUnsafe ? '请立即复核并处置风险人员' : summary ? '本次现场研判完成，未发现需立即处置的风险' : '系统正在等待或处理新的现场图像' }}</strong></div><div class="task-meta">任务状态：{{ statusLabel(task?.status) }}　·　{{ task?.createTime || '—' }}</div></section>
  </main>
</template>

<script setup>
import { computed, defineComponent, h, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { getInferenceTask, getInferenceTaskList, getTaskHazards, getTaskSceneGraph } from '@/api/infer'

const router = useRouter()
const task = ref(null), image = ref(null), summary = ref(null)
const objects = ref([]), relations = ref([]), hazards = ref([])
const graphZoom = ref(1)
const clock = reactive({ time: '--:--:--', date: '', week: '' })
const steps = [{ key: 'detector', label: '目标检测' }, { key: 'scene_graph', label: '场景图构建' }, { key: 'reasoning', label: '安全研判' }]
let dataTimer, clockTimer

const Empty = defineComponent({ props: { text: String }, setup: props => () => h('div', { class: 'empty' }, [h('span', '▧'), h('p', props.text)]) })
const PanelTitle = defineComponent({ props: { no: String, title: String, subtitle: String }, setup: props => () => h('header', { class: 'panel-title' }, [h('span', props.no), h('div', [h('h2', props.title), h('p', props.subtitle)])]) })
const imageUrl = computed(() => image.value?.fileUrl || '')
const canvasWidth = computed(() => Number(image.value?.width) || 1280)
const canvasHeight = computed(() => Number(image.value?.height) || 720)
const graphTransform = computed(() => {
  const x = canvasWidth.value / 2
  const y = canvasHeight.value / 2
  return `translate(${x} ${y}) scale(${graphZoom.value}) translate(${-x} ${-y})`
})
const hasUnsafe = computed(() => Boolean(summary.value?.hasUnsafeBehavior))
const riskTitle = computed(() => summary.value?.highestRiskCategory || (summary.value ? '未发现不安全行为' : '待生成'))
const riskLevel = computed(() => !summary.value ? '待判定' : hasUnsafe.value ? '需处置' : '安全')
const recommendation = computed(() => {
  if (!summary.value) return '等待系统完成场景关系分析与安全研判。'
  if (!hasUnsafe.value) return '现场状态正常，请继续保持个人防护用品规范佩戴。'
  const missing = hazards.value.flatMap(item => item.missingPpeJson || [])
  return missing.length ? `请立即核查并补齐：${[...new Set(missing)].join('、')}；完成后复检。` : '请立即核查风险人员作业状态，并按现场安全规程完成整改后复检。'
})

const box = object => object.bbox || [0, 0, 0, 0]
const boxWidth = object => Math.max(0, box(object)[2] - box(object)[0])
const boxHeight = object => Math.max(0, box(object)[3] - box(object)[1])
const score = object => `${Math.round((object.score || 0) * 100)}%`
function objectTone(object) { const label = object.labelName || ''; if (/worker|person|工人|人员/i.test(label)) return 'worker'; if (/helmet|vest|glove|安全帽|反光|手套/i.test(label)) return 'ppe'; if (/scaffold|edge|ladder|脚手架|边缘/i.test(label)) return 'risk'; return 'device' }
function objectColor(object) { return ({ worker: '#ff8d2c', ppe: '#b8f257', risk: '#ff6464', device: '#57d6e8' })[objectTone(object)] }
function zoomGraph(event) {
  const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18
  graphZoom.value = Math.min(4, Math.max(0.7, Number((graphZoom.value * factor).toFixed(2))))
}
function resetGraphZoom() { graphZoom.value = 1 }
const nodeX = object => (box(object)[0] + box(object)[2]) / 2
const nodeY = object => (box(object)[1] + box(object)[3]) / 2
const nodeRadius = object => objectTone(object) === 'worker' ? 15 : 11
const fromNode = rel => objects.value.find(object => object.id === rel.subjectObjectId)
const toNode = rel => objects.value.find(object => object.id === rel.objectObjectId)
const isHazardObject = object => hazards.value.some(item => item.workerObjectId === object.id && item.predMajorLabel !== 'safe')
const statusLabel = status => ({ pending: '排队中', running: '研判中', completed: '已完成', failed: '失败' })[status] || '等待中'
function stepStatus(key) { if (!task.value) return ''; const order = steps.findIndex(step => step.key === key); const current = steps.findIndex(step => step.key === task.value.currentStage); if (task.value.status === 'completed' || current > order) return 'done'; if (current === order && task.value.status === 'running') return 'active'; return '' }

async function loadTask(taskId) {
  const isNewTask = task.value?.id !== taskId
  const [detail, graph, hazard] = await Promise.all([getInferenceTask(taskId), getTaskSceneGraph(taskId), getTaskHazards(taskId)])
  task.value = detail.task; image.value = detail.image; summary.value = detail.summary || hazard.imageLevelResult
  objects.value = graph.objects || []; relations.value = graph.relationships || []; hazards.value = hazard.workerResults || []
  if (isNewTask) resetGraphZoom()
}
async function load() {
  try {
    // 大屏必须以任务表为唯一数据源。不要从统计接口中按状态挑选，
    // 否则旧的 running 记录会长期遮蔽最新完成任务。
    const { rows } = await getInferenceTaskList({ pageNum: 1, pageSize: 1 })
    const latestTask = rows?.[0]
    if (latestTask) await loadTask(latestTask.id)
  } catch { /* preserve the last operational frame */ }
}
function tick() { const date = new Date(), pad = value => String(value).padStart(2, '0'); clock.time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`; clock.date = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`; clock.week = `星期${'日一二三四五六'[date.getDay()]}` }
onMounted(() => { tick(); clockTimer = setInterval(tick, 1000); load(); dataTimer = setInterval(load, 3000) })
onBeforeUnmount(() => { clearInterval(dataTimer); clearInterval(clockTimer) })
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
.screen{--bg:#070d14;--panel:#0d1723;--line:#234157;--muted:#7990a5;--text:#e7f0f8;--orange:#ff7518;--cyan:#46c6dc;--green:#b7f04a;--red:#ff514c;position:fixed;inset:0;z-index:3000;display:flex;flex-direction:column;gap:12px;padding:16px 20px;background:radial-gradient(900px 460px at 50% -10%,rgba(39,114,147,.16),transparent 65%),#070d14;color:var(--text);font-family:Rajdhani,'Microsoft YaHei',sans-serif;overflow:hidden}.screen:before{content:'';position:absolute;inset:0;pointer-events:none;opacity:.35;background-image:linear-gradient(rgba(90,154,180,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(90,154,180,.055) 1px,transparent 1px);background-size:42px 42px}.screen>*{position:relative;z-index:1}.topbar,.taskbar,.bottomline{border:1px solid var(--line);background:linear-gradient(180deg,rgba(14,26,40,.95),rgba(8,14,23,.94));border-radius:4px}.topbar{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;border-left:5px solid var(--orange)}.brand{display:flex;gap:13px;align-items:center}.brand-mark{display:grid;place-items:center;width:42px;height:42px;border:1px solid rgba(255,117,24,.6);color:var(--orange);font-size:23px;background:rgba(255,117,24,.08)}.brand h1{margin:0;font-size:25px;letter-spacing:3px}.brand p{margin:4px 0 0;color:#8da7bf;font:10px 'JetBrains Mono';letter-spacing:2.5px}.topbar-right{display:flex;gap:20px;align-items:center}.live{padding:6px 12px;color:var(--green);border:1px solid rgba(183,240,74,.36);background:rgba(183,240,74,.06);font-size:13px}.live i,.result-summary i{display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}.clock{text-align:right}.clock b{display:block;font:700 27px 'JetBrains Mono';letter-spacing:2px}.clock span{color:var(--muted);font-size:11px}.topbar button{border:1px solid var(--line);background:transparent;color:#a7bbce;padding:8px 13px;cursor:pointer;font-family:inherit}.taskbar{display:flex;justify-content:space-between;align-items:center;padding:8px 16px;font-size:13px}.taskbar>div:first-child{display:flex;align-items:center;gap:11px}.taskbar span{color:var(--muted)}.taskbar strong{font-size:15px}.taskbar code{color:var(--cyan);font:11px 'JetBrains Mono'}.task-steps{display:flex;gap:22px}.task-steps span{display:flex;align-items:center;gap:6px}.task-steps i{width:8px;height:8px;border-radius:50%;background:#405367}.task-steps .done{color:var(--green)}.task-steps .done i{background:var(--green)}.task-steps .active{color:#ffbd4a}.task-steps .active i{background:#ffbd4a;box-shadow:0 0 9px #ffbd4a;animation:pulse 1s infinite}.four-stage{flex:1;min-height:0;display:grid;grid-template-columns:1fr 1fr 1fr 1.05fr;gap:12px}.panel{min-width:0;min-height:0;display:flex;flex-direction:column;padding:12px;border:1px solid var(--line);background:linear-gradient(180deg,rgba(13,23,35,.97),rgba(8,15,24,.97));position:relative}.panel:before{content:'';position:absolute;left:-1px;top:-1px;width:16px;height:16px;border-left:2px solid var(--orange);border-top:2px solid var(--orange)}.panel-title{display:flex;gap:9px;align-items:flex-start;margin-bottom:10px}.panel-title>span{color:var(--orange);font:700 12px 'JetBrains Mono';padding-top:2px}.panel-title h2{margin:0;font-size:17px;letter-spacing:1px}.panel-title p{margin:2px 0 0;color:var(--muted);font-size:11px}.media-stage{position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#050a10;border:1px solid #1c3346}.media-stage>img,.image-svg,.graph-svg{width:100%;height:100%;object-fit:contain}.image-svg,.graph-svg{display:block}.bbox{fill:rgba(0,0,0,.04);stroke-width:2}.bbox.worker{stroke:#ff8d2c}.bbox.ppe{stroke:#9eea3e}.bbox.risk{stroke:#ff5353}.bbox.device{stroke:#49cbe0}.box-label{font:600 12px 'JetBrains Mono';paint-order:stroke;stroke:#07101a;stroke-width:3px;stroke-linejoin:round}.box-label.worker{fill:#ff9d47}.box-label.ppe{fill:#b9f35b}.box-label.risk{fill:#ff6969}.box-label.device{fill:#63d9ea}.graph-stage{background:radial-gradient(circle at 50% 50%,rgba(38,96,126,.14),transparent 68%),#07111a}.edge{stroke:#66849e;stroke-width:1.4;opacity:.8}.edge-label{fill:#b6c9d8;font:11px 'JetBrains Mono';paint-order:stroke;stroke:#07111a;stroke-width:3px}.node{stroke-width:2;fill:#315672;stroke:#8ab3cc}.node.worker{fill:#a84b17;stroke:#ff9b4a}.node.ppe{fill:#577c28;stroke:#b8f257}.node.risk{fill:#a43337;stroke:#ff6d72}.node.device{fill:#20687b;stroke:#57d6e8}.node.hazard{stroke:#ff3d3d;stroke-width:4}.node-label{fill:#e1edf6;font:11px 'Microsoft YaHei';text-anchor:middle;paint-order:stroke;stroke:#07111a;stroke-width:3px}.panel footer{display:flex;justify-content:space-between;gap:8px;min-height:27px;padding-top:9px;color:var(--muted);font-size:10px;white-space:nowrap;overflow:hidden}.badge{padding:2px 5px;border:1px solid;font:10px 'JetBrains Mono'}.badge.blue{color:#6bd8ec;border-color:#306d80}.badge.teal{color:#a2ecca;border-color:#2f7968}.empty{display:flex;flex-direction:column;align-items:center;gap:8px;color:#5f788e}.empty span{font-size:36px}.empty p{margin:0;font-size:13px}.result-panel{border-color:#426078}.verdict{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:166px;border:1px solid #31475c;background:rgba(75,123,181,.04);text-align:center}.verdict-icon{display:grid;place-items:center;width:48px;height:48px;margin-bottom:8px;border-radius:50%;border:2px solid #849aab;color:#b8cad7;font:700 27px 'JetBrains Mono'}.verdict p{margin:0;color:#b7c8d7;font-size:16px}.verdict strong{max-width:90%;margin-top:8px;color:#fff;font-size:20px;overflow-wrap:anywhere}.verdict.safe{border-color:rgba(183,240,74,.38);background:rgba(183,240,74,.05)}.verdict.safe .verdict-icon,.verdict.safe strong{border-color:var(--green);color:var(--green)}.verdict.danger{border-color:rgba(255,81,76,.58);background:rgba(255,81,76,.08);animation:danger 1.6s infinite}.verdict.danger .verdict-icon,.verdict.danger strong{border-color:var(--red);color:var(--red)}.result-details{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.result-details div{padding:9px;border:1px solid #263c4e;background:#0a131e}.result-details span,.result-details b{display:block}.result-details span{color:var(--muted);font-size:11px}.result-details b{margin-top:3px;font-size:17px}.red{color:var(--red)!important}.recommendation{margin-top:auto;padding:11px;border-left:3px solid var(--orange);background:rgba(255,117,24,.06)}.recommendation span{color:#ffae76;font-size:11px}.recommendation p{margin:5px 0 0;color:#cedce8;font-size:13px;line-height:1.55}.bottomline{display:flex;align-items:center;justify-content:space-between;padding:10px 16px}.result-summary{font-size:15px}.result-summary i.red{background:var(--red);box-shadow:0 0 8px var(--red)}.task-meta{color:var(--muted);font:10px 'JetBrains Mono'}@keyframes pulse{50%{opacity:.35}}@keyframes danger{50%{box-shadow:inset 0 0 24px rgba(255,81,76,.12)}}@media(max-width:1450px){.screen{padding:10px}.brand h1{font-size:20px}.four-stage{gap:8px}.panel{padding:9px}.panel-title h2{font-size:15px}.topbar-right{gap:10px}.clock b{font-size:21px}.recommendation p{font-size:11px}}@media(max-width:1050px){.screen{position:relative;min-height:100vh;overflow:auto}.four-stage{grid-template-columns:repeat(2,minmax(320px,1fr));min-height:900px}.topbar{align-items:flex-start}.taskbar,.bottomline{flex-wrap:wrap;gap:8px}}
.graph-stage{cursor:zoom-in}.graph-stage:active{cursor:zoom-out}.zoom-tip{position:absolute;right:8px;bottom:8px;padding:4px 6px;color:#8ea7ba;border:1px solid rgba(74,125,153,.45);background:rgba(4,12,20,.78);font:9px 'JetBrains Mono';pointer-events:none}
</style>
