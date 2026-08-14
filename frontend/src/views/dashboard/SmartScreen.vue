<template>
  <section class="smart-screen">
    <div class="bg-grid"></div><div class="bg-radial"></div><div class="scan-sweep"></div>

    <!-- ═══ HEADER ═══ -->
    <header class="ss-header">
      <div class="header-left">
        <div class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/><line x1="12" y1="22" x2="12" y2="15.5"/><polyline points="22 8.5 12 15.5 2 8.5"/></svg></div>
        <div><div class="brand-title">SAFESCENE</div><div class="brand-sub">CONSTRUCTION SAFETY AI PLATFORM</div></div>
      </div>
      <div class="header-center">
        <div class="header-stat" v-for="m in headerMetrics" :key="m.key">
          <span class="val" :class="m.cls">{{ formatNum(stats[m.key] || 0) }}</span>
          <span class="lbl">{{ m.label }}</span>
        </div>
      </div>
      <div class="header-right">
        <div class="sys-status"><span class="status-dot"></span><span>SYSTEM ONLINE</span></div>
        <div><div class="clock">{{ clockTime }}</div><div class="clock-date">{{ clockDate }}</div></div>
      </div>
    </header>

    <!-- ═══ MAIN ═══ -->
    <div class="ss-main">
      <!-- ── LEFT TOP: Image Gallery ── -->
      <div class="panel">
        <div class="corner-deco tl"></div><div class="corner-deco tr"></div><div class="corner-deco bl"></div><div class="corner-deco br"></div>
        <div class="panel-head">
          <div><div class="panel-kicker">SITE SURVEILLANCE</div><div class="panel-title">施工现场图像监控</div></div>
          <span class="panel-badge badge-cyan">{{ galleryImages.length }} IMAGES</span>
        </div>
        <div class="panel-body">
          <div class="image-grid">
            <div v-for="(img, idx) in galleryImages" :key="img.id"
              class="image-thumb" :class="{ active: selectedImageId === img.id }"
              @click="selectedImageId = img.id">
              <img v-if="img.thumbUrl" :src="img.thumbUrl" class="thumb-img" />
              <div v-else class="thumb-placeholder" :class="'site-' + img.colorClass"></div>
              <span class="thumb-status" :class="img.status">{{ statusText(img.status) }}</span>
              <div class="thumb-info"><span>#{{ img.id }}</span><span>{{ img.time }}</span></div>
            </div>
            <div v-if="!galleryImages.length" class="empty-gallery">NO IMAGES</div>
          </div>
        </div>
      </div>

      <!-- ── CENTER: KPIs + Pipeline + Scene Graph ── -->
      <div class="panel center-panel">
        <div class="corner-deco tl"></div><div class="corner-deco tr"></div><div class="corner-deco bl"></div><div class="corner-deco br"></div>
        <div class="panel-head">
          <div><div class="panel-kicker">SCENE GRAPH · TASK #{{ selectedImageId || '—' }}</div><div class="panel-title">施工场景图实时构建</div></div>
          <span class="panel-badge" :class="activeTask ? 'badge-amber' : 'badge-cyan'">{{ activeTask ? 'LIVE' : 'STANDBY' }}</span>
        </div>
        <div class="panel-body">
          <div class="kpi-row">
            <div class="kpi-card k-cyan"><div class="kpi-label">检测目标</div><div class="kpi-val">{{ stats.totalObjects || 0 }}</div><div class="kpi-sub">YOLOv12x</div></div>
            <div class="kpi-card k-teal"><div class="kpi-label">语义关系</div><div class="kpi-val">{{ stats.totalRelations || 0 }}</div><div class="kpi-sub">HTCL · mR@50</div></div>
            <div class="kpi-card k-amber"><div class="kpi-label">推理任务</div><div class="kpi-val">{{ stats.totalInferenceTasks || 0 }}</div><div class="kpi-sub">{{ stats.runningTasks || 0 }} running</div></div>
            <div class="kpi-card k-red"><div class="kpi-label">风险现场</div><div class="kpi-val">{{ stats.unsafeImages || 0 }}</div><div class="kpi-sub">PPE 缺失 / 距离违规</div></div>
          </div>
          <div class="pipeline-track">
            <div v-for="(stage, idx) in stages" :key="stage.key"
              class="pipeline-stage" :class="stageCls(stage, idx)">
              <div class="stage-idx">
                <svg v-if="stageState(stage,idx)==='done'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                <span v-else>{{ idx + 1 }}</span>
              </div>
              <div class="stage-info"><b>{{ stage.label }}</b><small>{{ stageDesc(stage, idx) }}</small></div>
            </div>
          </div>
          <div class="scene-graph-area">
            <svg class="sg-canvas" viewBox="0 0 1060 520">
              <text v-if="!hasSceneData" x="530" y="260" text-anchor="middle" fill="#3d5570" font-size="13" font-family="Share Tech Mono">AWAITING SCENE DATA...</text>
            </svg>
            <div class="sg-legend">
              <span><i style="background:#ff6b35"></i>工人</span>
              <span><i style="background:#00ff88"></i>安全设施</span>
              <span><i style="background:#ff3b5c"></i>危险结构</span>
              <span><i style="background:#4a90d9"></i>机械设备</span>
              <span><i style="background:#ffaa00"></i>作业区域</span>
            </div>
            <div class="sg-stats-overlay">
              <div class="sg-stat-item"><span class="v">{{ sceneNodes }}</span><span class="l">ENTITIES</span></div>
              <div class="sg-stat-item"><span class="v">{{ stats.totalRelations || 0 }}</span><span class="l">RELATIONS</span></div>
              <div class="sg-stat-item"><span class="v">{{ stats.unsafeImages || 0 }}</span><span class="l">HAZARDS</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── RIGHT TOP: Inference Results ── -->
      <div class="panel">
        <div class="corner-deco tl"></div><div class="corner-deco tr"></div><div class="corner-deco bl"></div><div class="corner-deco br"></div>
        <div class="panel-head">
          <div><div class="panel-kicker">INFERENCE RESULTS</div><div class="panel-title">不安全行为推理结果</div></div>
          <span class="panel-badge badge-red">{{ hazardCount }} ALERTS</span>
        </div>
        <div class="panel-body">
          <div class="hazard-cards">
            <div v-for="(w, idx) in workerResults" :key="idx"
              class="hz-card" :class="isUnsafe(w) ? 'unsafe' : 'safe'">
              <div class="hz-head">
                <div class="hz-worker"><span class="wi">{{ idx + 1 }}</span> Worker-{{ idx + 1 }}</div>
                <span class="hz-tag" :class="isUnsafe(w) ? 'danger' : 'ok'">{{ behaviorLabel(w) }}</span>
              </div>
              <div class="hz-body">
                <div class="hz-row" v-if="w.contextText"><span class="rk">场景上下文</span><span class="rv">{{ w.contextText }}</span></div>
                <div class="hz-row" v-if="w.relationType"><span class="rk">触发关系</span><span class="rv">{{ w.relationType }}</span></div>
                <div class="hz-ppe" v-if="w.missingPpeJson?.length || w.presentPpeJson?.length">
                  <span v-for="ppe in (w.missingPpeJson||[])" :key="'m'+ppe" class="ppe-chip miss">✕ {{ ppe }}</span>
                  <span v-for="ppe in (w.presentPpeJson||[])" :key="'p'+ppe" class="ppe-chip have">✓ {{ ppe }}</span>
                </div>
                <div class="hz-conf">
                  <div class="hz-conf-bar"><i :style="{ width: ((w.predScore||0)*100)+'%', background: isUnsafe(w) ? 'var(--red)' : 'var(--green)' }"></i></div>
                  <span class="hz-conf-val" :style="{ color: isUnsafe(w) ? 'var(--red)' : 'var(--green)' }">{{ ((w.predScore||0)*100).toFixed(1) }}%</span>
                </div>
              </div>
            </div>
            <div v-if="!workerResults.length" class="empty-results">NO INFERENCE RESULTS</div>
          </div>
        </div>
      </div>

      <!-- ── LEFT BOTTOM: Detection Preview ── -->
      <div class="panel">
        <div class="corner-deco tl"></div><div class="corner-deco tr"></div><div class="corner-deco bl"></div><div class="corner-deco br"></div>
        <div class="panel-head">
          <div><div class="panel-kicker">DETECTION VIEW</div><div class="panel-title">当前图像检测结果</div></div>
          <span class="panel-badge badge-green">#{{ selectedImageId || '—' }}</span>
        </div>
        <div class="panel-body">
          <div class="detection-preview">
            <div class="det-image-wrap">
              <img v-if="selectedImageUrl" :src="selectedImageUrl" style="width:100%;height:100%;object-fit:cover;" />
              <div v-else class="thumb-placeholder site-a" style="position:absolute;inset:0"></div>
              <div class="det-overlay-stats">
                <div class="det-stat"><span class="dv">{{ stats.totalObjects || 0 }}</span><span class="dl">OBJECTS</span></div>
                <div class="det-stat"><span class="dv">{{ stats.totalRelations || 0 }}</span><span class="dl">RELATIONS</span></div>
                <div class="det-stat"><span class="dv" style="color:var(--red)">{{ stats.unsafeImages || 0 }}</span><span class="dl">HAZARDS</span></div>
              </div>
            </div>
            <div class="det-meta-bar">
              <span class="active-file">Scene #{{ selectedImageId || '—' }}</span>
              <span>YOLOv12x → HTCL → Dual-Encoder</span>
            </div>
          </div>
        </div>
      </div>

      <!-- ── RIGHT BOTTOM: Risk + Tasks ── -->
      <div class="panel">
        <div class="corner-deco tl"></div><div class="corner-deco tr"></div><div class="corner-deco bl"></div><div class="corner-deco br"></div>
        <div class="panel-head">
          <div><div class="panel-kicker">RISK MONITOR</div><div class="panel-title">风险态势 & 任务队列</div></div>
          <span class="panel-badge badge-amber">LIVE</span>
        </div>
        <div class="panel-body">
          <div class="risk-gauge">
            <div class="gauge-ring">
              <svg viewBox="0 0 80 80"><circle class="gauge-bg" cx="40" cy="40" r="34"/><circle class="gauge-fill" cx="40" cy="40" r="34" :stroke-dasharray="gaugeCircum" :stroke-dashoffset="gaugeOffset"/></svg>
              <div class="gauge-center"><span class="gauge-val">{{ riskRate }}</span><span class="gauge-unit">% RISK</span></div>
            </div>
            <div class="gauge-info">
              <h4>风险图像占比</h4>
              <div class="gauge-breakdown">
                <div class="gauge-row"><i style="background:#ff3b5c"></i><span>高风险现场</span><span class="rv">{{ stats.unsafeImages || 0 }}</span></div>
                <div class="gauge-row"><i style="background:#00ff88"></i><span>安全完成</span><span class="rv">{{ stats.safeImages || 0 }}</span></div>
                <div class="gauge-row"><i style="background:#3d5570"></i><span>待处理</span><span class="rv">{{ pendingCount }}</span></div>
              </div>
            </div>
          </div>
          <div class="task-queue">
            <div v-for="(task, idx) in recentTasks.slice(0, 6)" :key="task.id"
              class="task-item" :class="task.status">
              <div class="task-num">{{ String(idx + 1).padStart(2, '0') }}</div>
              <div class="task-meta"><b>{{ pipelineLabel(task.pipelineType) }}</b><small>{{ task.createTime || '—' }}</small></div>
              <span class="task-status" :class="'s-' + task.status">{{ statusLabel(task.status) }}</span>
            </div>
            <div v-if="!recentTasks.length" class="empty-tq">NO TASKS YET</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { getDashboardStats } from '@/api/dashboard'

const stats = ref({})
const recentTasks = ref([])
const workerResults = ref([])
const clockTime = ref('--:--:--')
const clockDate = ref('')
const selectedImageId = ref(null)
let refreshTimer, clockTimer

const stages = [
  { key: 'detector', label: '目标检测' },
  { key: 'scene_graph', label: '场景图构建' },
  { key: 'reasoning', label: '危险推理' },
]
const headerMetrics = [
  { key: 'totalImages', label: '现场图像', cls: 'c-cyan' },
  { key: 'totalObjects', label: '检测实体', cls: 'c-green' },
  { key: 'totalRelations', label: '语义关系', cls: 'c-cyan' },
  { key: 'totalInferenceTasks', label: '推理任务', cls: 'c-amber' },
  { key: 'unsafeImages', label: '风险现场', cls: 'c-red' },
]
const colorClasses = ['site-a','site-b','site-c','site-d','site-e','site-f']

const activeTask = computed(() => recentTasks.value.find(t => t.status === 'running') || recentTasks.value.find(t => t.status === 'pending') || null)
const pendingCount = computed(() => recentTasks.value.filter(t => t.status === 'pending').length)
const riskRate = computed(() => { const t = Number(stats.value.totalSummaries || 0); return t ? Math.round((Number(stats.value.unsafeImages || 0) / t) * 100) : 0 })
const gaugeCircum = 2 * Math.PI * 34
const gaugeOffset = computed(() => gaugeCircum * (1 - riskRate.value / 100))
const hasSceneData = computed(() => (stats.value.totalObjects || 0) > 0)
const sceneNodes = computed(() => Math.min(stats.value.totalObjects || 0, 24))
const hazardCount = computed(() => workerResults.value.filter(w => isUnsafe(w)).length)

const galleryImages = computed(() => {
  return recentTasks.value.slice(0, 6).map((t, i) => ({
    id: t.id,
    status: t.status === 'completed' ? (t.hasUnsafe ? 'unsafe' : 'safe') : t.status,
    time: (t.createTime || '').slice(11, 19) || '—',
    colorClass: colorClasses[i % 6],
    thumbUrl: t.imageUrl || null,
  }))
})
const selectedImageUrl = computed(() => {
  const img = galleryImages.value.find(g => g.id === selectedImageId.value)
  return img?.thumbUrl || null
})

const formatNum = n => Number(n || 0).toLocaleString()
const pipelineLabel = t => ({ full_pipeline: '全流程推理', scene_graph_only: '场景图推理', reasoning_only: '风险推理' })[t] || t || '推理任务'
const statusLabel = s => ({ pending: 'PENDING', running: 'RUNNING', completed: 'DONE', failed: 'FAIL' })[s] || s
const statusText = s => ({ safe: 'SAFE', unsafe: 'UNSAFE', pending: 'PENDING', running: 'RUNNING', completed: 'DONE', failed: 'FAIL' })[s] || s
const isUnsafe = w => w.unsafeBehaviorCategory && w.unsafeBehaviorCategory !== 'safe'
const categoryMap = { climbing_scaffold_frame: '攀爬脚手架', leaning_out: '身体探出', standing_on_guardrail: '站立护栏', throwing_material: '抛掷物料', unsafe_posture: '不安全姿态', no_helmet: '未戴安全帽', no_harness: '未系安全带', no_vest: '未穿反光衣', excavation_proximity: '挖掘接近', scaffold_work: '脚手架作业', general_work: '常规作业', mechanical_operation: '机械操作' }
const behaviorLabel = w => isUnsafe(w) ? (categoryMap[w.unsafeBehaviorCategory] || w.unsafeBehaviorCategory) : '安全'

function stageState(stage, idx) {
  if (!activeTask.value) return 'idle'
  const ci = stages.findIndex(s => s.key === activeTask.value.currentStage)
  if (ci > idx) return 'done'; if (ci === idx) return 'active'; return 'idle'
}
function stageCls(stage, idx) { const s = stageState(stage, idx); return s === 'done' ? 'done' : s === 'active' ? 'active' : '' }
function stageDesc(stage, idx) { const s = stageState(stage, idx); return s === 'done' ? 'Completed' : s === 'active' ? 'Processing...' : 'Waiting' }

function updateClock() {
  const n = new Date()
  clockTime.value = [n.getHours(), n.getMinutes(), n.getSeconds()].map(v => String(v).padStart(2,'0')).join(':')
  const d = ['SUN','MON','TUE','WED','THU','FRI','SAT']
  clockDate.value = `${n.getFullYear()}/${String(n.getMonth()+1).padStart(2,'0')}/${String(n.getDate()).padStart(2,'0')} ${d[n.getDay()]}`
}

async function refreshDashboard() {
  try {
    const data = await getDashboardStats()
    stats.value = data || {}
    recentTasks.value = data?.recentTasks || []
    if (selectedImageId.value == null && recentTasks.value.length) selectedImageId.value = recentTasks.value[0].id
  } catch { /* keep last good view */ }
}

onMounted(async () => { updateClock(); clockTimer = setInterval(updateClock, 1000); await refreshDashboard(); refreshTimer = setInterval(refreshDashboard, 5000) })
onBeforeUnmount(() => { clearInterval(refreshTimer); clearInterval(clockTimer) })
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');
.smart-screen { --bg-void:#040810;--bg-panel:rgba(10,22,48,0.75);--border-glow:rgba(0,200,255,0.12);--border-line:rgba(0,200,255,0.06);--cyan:#00d4ff;--cyan-dim:#0098c2;--cyan-glow:rgba(0,212,255,0.15);--teal:#00ffc8;--amber:#ffaa00;--red:#ff3b5c;--green:#00ff88;--text-bright:#e4eef8;--text-mid:#7a96b8;--text-dim:#3d5570;--font-display:'Chakra Petch','Microsoft YaHei',sans-serif;--font-body:'Rajdhani','Microsoft YaHei',sans-serif;--font-mono:'Share Tech Mono','Consolas',monospace; position:relative;width:100%;height:100vh;min-width:1400px;display:flex;flex-direction:column;background:var(--bg-void);color:var(--text-bright);font-family:var(--font-body);overflow:hidden; }
.bg-grid{position:absolute;inset:0;z-index:0;pointer-events:none;background:linear-gradient(rgba(0,200,255,0.015) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,255,0.015) 1px,transparent 1px);background-size:60px 60px;animation:gridShift 20s linear infinite}@keyframes gridShift{to{background-position:60px 60px}}.bg-radial{position:absolute;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse 900px 500px at 50% 45%,rgba(0,180,255,0.04) 0%,transparent 70%)}.scan-sweep{position:absolute;top:0;left:0;right:0;height:2px;z-index:1;pointer-events:none;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:0.3;animation:sweepDown 8s ease-in-out infinite}@keyframes sweepDown{0%,100%{top:0;opacity:0}10%{opacity:0.3}90%{opacity:0.3}95%{opacity:0}100%{top:100%}}
.ss-header{height:64px;flex-shrink:0;display:flex;align-items:center;justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--border-glow);background:linear-gradient(180deg,rgba(0,20,50,0.6) 0%,transparent 100%);position:relative;z-index:2}.ss-header::after{content:'';position:absolute;bottom:-1px;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:0.4}.header-left,.header-center,.header-right{display:flex;align-items:center;gap:14px}.brand-mark{width:38px;height:38px;display:flex;align-items:center;justify-content:center;border:1px solid var(--cyan);border-radius:8px;background:rgba(0,200,255,0.06)}.brand-mark svg{width:22px;height:22px;color:var(--cyan)}.brand-title{font-family:var(--font-display);font-size:20px;font-weight:700;letter-spacing:2px;background:linear-gradient(135deg,var(--cyan),#60e0ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.brand-sub{font-size:10px;color:var(--text-dim);letter-spacing:3px;font-family:var(--font-mono)}.header-stat{display:flex;flex-direction:column;align-items:center;padding:0 16px;border-right:1px solid var(--border-line)}.header-stat:last-child{border-right:none}.header-stat .val{font-family:var(--font-mono);font-size:18px;font-weight:700;line-height:1.1}.c-cyan{color:var(--cyan);text-shadow:0 0 12px rgba(0,212,255,0.3)}.c-green{color:var(--green);text-shadow:0 0 12px rgba(0,255,136,0.3)}.c-amber{color:var(--amber);text-shadow:0 0 12px rgba(255,170,0,0.3)}.c-red{color:var(--red);text-shadow:0 0 12px rgba(255,59,92,0.3)}.header-stat .lbl{font-size:9px;color:var(--text-dim);letter-spacing:1px;font-family:var(--font-mono)}.sys-status{display:flex;align-items:center;gap:8px;font-family:var(--font-mono);font-size:10px;color:var(--text-mid)}.status-dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s ease-in-out infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}.clock{font-family:var(--font-mono);font-size:18px;font-weight:600;color:var(--text-mid);letter-spacing:2px}.clock-date{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);letter-spacing:1px;text-align:right}
.ss-main{flex:1;display:grid;grid-template-columns:400px 1fr 400px;grid-template-rows:1fr 1fr;gap:10px;padding:10px 14px 14px;min-height:0;position:relative;z-index:2}.panel{background:var(--bg-panel);border:1px solid var(--border-glow);border-radius:6px;position:relative;overflow:hidden;display:flex;flex-direction:column;animation:fadeInUp 0.5s ease both}.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:0.5}.panel::after{content:'';position:absolute;inset:0;pointer-events:none;background:linear-gradient(180deg,rgba(0,180,255,0.02) 0%,transparent 30%);border-radius:6px}@keyframes fadeInUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}.ss-main .panel:nth-child(1){animation-delay:.08s}.ss-main .panel:nth-child(2){animation-delay:.12s}.ss-main .panel:nth-child(3){animation-delay:.16s}.ss-main .panel:nth-child(4){animation-delay:.2s}.ss-main .panel:nth-child(5){animation-delay:.24s}.panel-head{display:flex;align-items:center;justify-content:space-between;padding:10px 14px 8px;border-bottom:1px solid var(--border-line);position:relative;z-index:1;flex-shrink:0}.panel-kicker{font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--cyan-dim);text-transform:uppercase;margin-bottom:1px}.panel-title{font-family:var(--font-display);font-size:14px;font-weight:600;color:var(--text-bright)}.panel-badge{font-family:var(--font-mono);font-size:9px;padding:2px 8px;border-radius:3px;border:1px solid}.badge-cyan{color:var(--cyan);border-color:rgba(0,200,255,0.2);background:rgba(0,200,255,0.06)}.badge-amber{color:var(--amber);border-color:rgba(255,170,0,0.2);background:rgba(255,170,0,0.06)}.badge-red{color:var(--red);border-color:rgba(255,59,92,0.2);background:rgba(255,59,92,0.06)}.badge-green{color:var(--green);border-color:rgba(0,255,136,0.2);background:rgba(0,255,136,0.06)}.panel-body{flex:1;padding:10px 14px;position:relative;z-index:1;overflow:hidden;display:flex;flex-direction:column;min-height:0}.corner-deco{position:absolute;width:12px;height:12px;z-index:3}.corner-deco::before,.corner-deco::after{content:'';position:absolute;background:var(--cyan);opacity:0.4}.corner-deco.tl{top:-1px;left:-1px}.corner-deco.tl::before{width:12px;height:1px}.corner-deco.tl::after{width:1px;height:12px}.corner-deco.tr{top:-1px;right:-1px}.corner-deco.tr::before{width:12px;height:1px;right:0}.corner-deco.tr::after{width:1px;height:12px;right:0}.corner-deco.bl{bottom:-1px;left:-1px}.corner-deco.bl::before{width:12px;height:1px;bottom:0}.corner-deco.bl::after{width:1px;height:12px;bottom:0}.corner-deco.br{bottom:-1px;right:-1px}.corner-deco.br::before{width:12px;height:1px;bottom:0;right:0}.corner-deco.br::after{width:1px;height:12px;bottom:0;right:0}
.image-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;flex:1;overflow-y:auto;align-content:start}.image-thumb{position:relative;aspect-ratio:16/10;border-radius:4px;overflow:hidden;border:1px solid var(--border-line);cursor:pointer;transition:border-color 0.25s,box-shadow 0.25s}.image-thumb:hover{border-color:rgba(0,200,255,0.3);box-shadow:0 0 16px rgba(0,200,255,0.08)}.image-thumb.active{border-color:var(--cyan);box-shadow:0 0 20px rgba(0,200,255,0.12)}.thumb-img{width:100%;height:100%;object-fit:cover;display:block}.thumb-placeholder{width:100%;height:100%;position:relative;overflow:hidden}.site-a{background:linear-gradient(145deg,#1a2a1a 0%,#2a3520 30%,#4a5a30 50%,#8a7a50 70%,#6a6a5a 100%)}.site-b{background:linear-gradient(160deg,#2a2a3a 0%,#3a3a4a 25%,#5a5060 50%,#8a7a6a 75%,#6a5a4a 100%)}.site-c{background:linear-gradient(135deg,#3a3020 0%,#5a4a30 35%,#7a6a40 55%,#9a8a50 80%,#4a4a3a 100%)}.site-d{background:linear-gradient(150deg,#1a2a3a 0%,#2a3a4a 30%,#4a5a6a 55%,#6a7a8a 80%,#3a4a5a 100%)}.site-e{background:linear-gradient(140deg,#2a1a1a 0%,#4a3a2a 30%,#6a5a3a 55%,#8a7a5a 80%,#5a4a3a 100%)}.site-f{background:linear-gradient(155deg,#1a3a2a 0%,#2a4a3a 30%,#3a5a4a 55%,#5a7a6a 80%,#2a3a2a 100%)}.thumb-placeholder::before{content:'';position:absolute;inset:0;background:linear-gradient(rgba(0,200,255,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,255,0.02) 1px,transparent 1px);background-size:20px 20px}.thumb-placeholder::after{content:'';position:absolute;bottom:0;left:0;right:0;height:40%;background:linear-gradient(transparent,rgba(4,8,16,0.7))}.thumb-status{position:absolute;top:6px;right:6px;z-index:3;font-family:var(--font-mono);font-size:8px;padding:2px 6px;border-radius:2px;letter-spacing:0.5px;font-weight:600}.thumb-status.safe{background:rgba(0,255,136,0.15);color:var(--green);border:1px solid rgba(0,255,136,0.25)}.thumb-status.unsafe{background:rgba(255,59,92,0.15);color:var(--red);border:1px solid rgba(255,59,92,0.25);animation:alertPulse 2s ease infinite}@keyframes alertPulse{0%,100%{box-shadow:none}50%{box-shadow:0 0 8px rgba(255,59,92,0.3)}}.thumb-status.pending{background:rgba(255,170,0,0.12);color:var(--amber);border:1px solid rgba(255,170,0,0.2)}.thumb-status.running{background:rgba(255,170,0,0.12);color:var(--amber);border:1px solid rgba(255,170,0,0.2)}.thumb-status.completed{background:rgba(0,255,136,0.12);color:var(--green);border:1px solid rgba(0,255,136,0.2)}.thumb-status.failed{background:rgba(255,59,92,0.12);color:var(--red);border:1px solid rgba(255,59,92,0.2)}.thumb-info{position:absolute;bottom:0;left:0;right:0;z-index:3;padding:4px 8px;display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:8px;color:var(--text-mid)}.empty-gallery{text-align:center;padding:40px;color:var(--text-dim);font-family:var(--font-mono);font-size:11px;grid-column:span 2}
.center-panel{grid-row:span 2}.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px;flex-shrink:0}.kpi-card{padding:10px 12px;background:rgba(0,20,50,0.4);border:1px solid var(--border-line);border-radius:4px;position:relative;overflow:hidden}.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}.k-cyan::before{background:linear-gradient(90deg,var(--cyan),transparent)}.k-teal::before{background:linear-gradient(90deg,var(--teal),transparent)}.k-amber::before{background:linear-gradient(90deg,var(--amber),transparent)}.k-red::before{background:linear-gradient(90deg,var(--red),transparent)}.kpi-label{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);margin-bottom:2px}.kpi-val{font-family:var(--font-mono);font-size:24px;font-weight:700;line-height:1.1}.k-cyan .kpi-val{color:var(--cyan);text-shadow:0 0 10px rgba(0,212,255,0.2)}.k-teal .kpi-val{color:var(--teal);text-shadow:0 0 10px rgba(0,255,200,0.2)}.k-amber .kpi-val{color:var(--amber);text-shadow:0 0 10px rgba(255,170,0,0.2)}.k-red .kpi-val{color:var(--red);text-shadow:0 0 10px rgba(255,59,92,0.2)}.kpi-sub{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);margin-top:2px}.pipeline-track{display:flex;gap:0;margin-bottom:10px;flex-shrink:0}.pipeline-stage{flex:1;display:flex;align-items:center;gap:8px;padding:8px 10px;border:1px solid var(--border-line);background:rgba(0,20,50,0.3);transition:all 0.3s}.pipeline-stage:first-child{border-radius:4px 0 0 4px}.pipeline-stage:last-child{border-radius:0 4px 4px 0}.pipeline-stage.active{border-color:var(--cyan);background:rgba(0,200,255,0.04)}.pipeline-stage.done{border-color:rgba(0,255,136,0.2)}.stage-idx{width:26px;height:26px;flex-shrink:0;display:flex;align-items:center;justify-content:center;border-radius:50%;border:1px solid var(--text-dim);font-family:var(--font-mono);font-size:10px;color:var(--text-dim);transition:all 0.3s}.pipeline-stage.active .stage-idx{border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 8px var(--cyan-glow)}.pipeline-stage.done .stage-idx{border-color:var(--green);color:var(--green)}.stage-info b{display:block;font-family:var(--font-display);font-size:11px;font-weight:600;color:var(--text-bright)}.stage-info small{font-size:9px;color:var(--text-dim);font-family:var(--font-mono)}.scene-graph-area{flex:1;min-height:0;position:relative;border:1px solid var(--border-line);border-radius:4px;overflow:hidden;background:radial-gradient(ellipse at center,rgba(0,100,180,0.03) 0%,transparent 70%),linear-gradient(rgba(0,200,255,0.01) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,255,0.01) 1px,transparent 1px);background-size:100% 100%,40px 40px,40px 40px}.sg-canvas{width:100%;height:100%}.sg-legend{position:absolute;bottom:8px;left:10px;display:flex;gap:12px;font-family:var(--font-mono);font-size:9px;color:var(--text-dim)}.sg-legend span{display:flex;align-items:center;gap:4px}.sg-legend i{width:7px;height:7px;border-radius:50%;display:inline-block}.sg-stats-overlay{position:absolute;top:8px;right:10px;display:flex;flex-direction:column;gap:4px;font-family:var(--font-mono);font-size:10px}.sg-stat-item{display:flex;align-items:center;gap:6px;padding:3px 8px;background:rgba(4,8,16,0.7);border:1px solid var(--border-line);border-radius:3px;backdrop-filter:blur(8px)}.sg-stat-item .v{font-weight:700;color:var(--cyan)}.sg-stat-item .l{color:var(--text-dim);font-size:8px}
.hazard-cards{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding-right:2px}.hazard-cards::-webkit-scrollbar{width:3px}.hazard-cards::-webkit-scrollbar-thumb{background:rgba(0,200,255,0.15);border-radius:2px}.hz-card{padding:10px 12px;background:rgba(0,20,50,0.4);border:1px solid var(--border-line);border-radius:4px;position:relative;overflow:hidden}.hz-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%}.hz-card.unsafe::before{background:var(--red)}.hz-card.safe::before{background:var(--green)}.hz-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}.hz-worker{display:flex;align-items:center;gap:6px;font-family:var(--font-display);font-size:12px;font-weight:600;color:var(--text-bright)}.hz-worker .wi{width:20px;height:20px;display:flex;align-items:center;justify-content:center;background:rgba(0,200,255,0.06);border-radius:50%;border:1px solid rgba(0,200,255,0.12);font-size:10px}.hz-tag{font-family:var(--font-mono);font-size:8px;padding:2px 6px;border-radius:2px;font-weight:600}.hz-tag.danger{background:rgba(255,59,92,0.12);color:var(--red);border:1px solid rgba(255,59,92,0.2)}.hz-tag.ok{background:rgba(0,255,136,0.08);color:var(--green);border:1px solid rgba(0,255,136,0.15)}.hz-body{display:flex;flex-direction:column;gap:4px}.hz-row{display:flex;justify-content:space-between;align-items:center}.hz-row .rk{font-family:var(--font-mono);font-size:9px;color:var(--text-dim)}.hz-row .rv{font-family:var(--font-mono);font-size:10px;color:var(--text-mid);max-width:55%;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.hz-ppe{display:flex;gap:4px;flex-wrap:wrap;margin-top:4px}.ppe-chip{font-family:var(--font-mono);font-size:8px;padding:1px 5px;border-radius:2px}.ppe-chip.miss{color:var(--red);background:rgba(255,59,92,0.08);border:1px solid rgba(255,59,92,0.12)}.ppe-chip.have{color:var(--green);background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.1)}.hz-conf{display:flex;align-items:center;gap:6px;margin-top:2px}.hz-conf-bar{flex:1;height:3px;background:rgba(255,255,255,0.04);border-radius:2px;overflow:hidden}.hz-conf-bar i{display:block;height:100%;border-radius:2px}.hz-conf-val{font-family:var(--font-mono);font-size:10px;font-weight:600;min-width:36px;text-align:right}.empty-results{text-align:center;padding:40px;color:var(--text-dim);font-family:var(--font-mono);font-size:11px}
.detection-preview{flex:1;display:flex;flex-direction:column;min-height:0}.det-image-wrap{flex:1;position:relative;border-radius:4px;overflow:hidden;border:1px solid var(--border-line);min-height:0}.det-overlay-stats{position:absolute;bottom:0;left:0;right:0;padding:8px 12px;background:linear-gradient(transparent,rgba(4,8,16,0.85));display:flex;gap:16px;z-index:3}.det-stat{display:flex;flex-direction:column}.det-stat .dv{font-family:var(--font-mono);font-size:16px;font-weight:700;color:var(--cyan);text-shadow:0 0 8px rgba(0,212,255,0.3);line-height:1}.det-stat .dl{font-family:var(--font-mono);font-size:8px;color:var(--text-dim);letter-spacing:0.5px;margin-top:2px}.det-meta-bar{display:flex;justify-content:space-between;align-items:center;margin-top:8px;padding:6px 0;font-family:var(--font-mono);font-size:10px;color:var(--text-dim)}.active-file{color:var(--text-mid)}
.risk-gauge{display:flex;align-items:center;gap:14px;padding:8px 0;border-bottom:1px solid var(--border-line);margin-bottom:8px;flex-shrink:0}.gauge-ring{width:68px;height:68px;position:relative;flex-shrink:0}.gauge-ring svg{width:68px;height:68px;transform:rotate(-90deg)}.gauge-ring circle{fill:none;stroke-width:5;stroke-linecap:round}.gauge-bg{stroke:rgba(255,59,92,0.1)}.gauge-fill{stroke:var(--red);transition:stroke-dashoffset 1s ease}.gauge-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}.gauge-val{font-family:var(--font-mono);font-size:17px;font-weight:700;color:var(--red);text-shadow:0 0 10px rgba(255,59,92,0.3);line-height:1}.gauge-unit{font-size:8px;color:var(--text-dim);font-family:var(--font-mono)}.gauge-info h4{font-family:var(--font-display);font-size:11px;font-weight:600;color:var(--text-bright);margin-bottom:3px}.gauge-breakdown{display:flex;flex-direction:column;gap:2px}.gauge-row{display:flex;align-items:center;gap:5px;font-family:var(--font-mono);font-size:9px;color:var(--text-mid)}.gauge-row i{width:5px;height:5px;border-radius:50%;flex-shrink:0}.gauge-row .rv{color:var(--text-bright);font-weight:600;margin-left:auto}.task-queue{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px}.task-item{display:flex;align-items:center;gap:8px;padding:6px 10px;background:rgba(0,20,50,0.3);border:1px solid var(--border-line);border-radius:4px}.task-item.running{border-left:2px solid var(--amber)}.task-item.completed{border-left:2px solid var(--green)}.task-item.pending{border-left:2px solid var(--text-dim)}.task-num{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);width:20px;height:20px;display:flex;align-items:center;justify-content:center;background:rgba(0,200,255,0.04);border-radius:3px;flex-shrink:0}.task-meta{flex:1;min-width:0}.task-meta b{display:block;font-size:10px;font-weight:600;color:var(--text-bright);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.task-meta small{font-family:var(--font-mono);font-size:8px;color:var(--text-dim)}.task-status{font-family:var(--font-mono);font-size:8px;padding:1px 6px;border-radius:2px;flex-shrink:0}.s-running{color:var(--amber);background:rgba(255,170,0,0.08);border:1px solid rgba(255,170,0,0.15)}.s-completed{color:var(--green);background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.12)}.s-pending{color:var(--text-dim);background:rgba(255,255,255,0.02);border:1px solid var(--border-line)}.s-failed{color:var(--red);background:rgba(255,59,92,0.06);border:1px solid rgba(255,59,92,0.12)}.empty-tq{text-align:center;padding:20px;color:var(--text-dim);font-family:var(--font-mono);font-size:11px}
</style>
