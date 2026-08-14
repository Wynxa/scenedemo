<template>
  <div class="scene-graph-view">
    <div v-if="!objects.length" class="empty-state">
      <el-icon :size="48"><PictureFilled /></el-icon>
      <p>运行流水线后在此查看场景图</p>
    </div>
    <div v-else class="graph-layout">
      <svg
        ref="svgRef"
        class="graph-canvas"
        :viewBox="`0 0 ${svgW} ${svgH}`"
        @mousemove="onMouseMove"
        @mouseleave="hoveredNode = null"
      >
        <!-- Relationship edges -->
        <line
          v-for="rel in positionedRels"
          :key="'r-' + rel.id"
          :x1="rel.x1" :y1="rel.y1"
          :x2="rel.x2" :y2="rel.y2"
          :stroke="rel.isHazard ? '#ef4444' : '#cbd5e1'"
          :stroke-width="rel.isHazard ? 2 : 1"
          :stroke-dasharray="rel.score < 0.6 ? '4 2' : 'none'"
          :opacity="rel.isHazard ? 0.8 : 0.5"
        />
        <!-- Predicate labels -->
        <text
          v-for="rel in positionedRels.filter(r => r.labelVisible)"
          :key="'rl-' + rel.id"
          :x="(rel.x1 + rel.x2) / 2"
          :y="(rel.y1 + rel.y2) / 2 - 6"
          text-anchor="middle"
          font-size="10"
          fill="#94a3b8"
          font-family="'JetBrains Mono', monospace"
        >{{ rel.predicateName }}</text>

        <!-- Object nodes -->
        <g
          v-for="(obj, i) in positionedNodes"
          :key="'n-' + obj.id"
          :transform="`translate(${obj.x},${obj.y})`"
          class="graph-node"
          @mouseenter="hoveredNode = obj"
        >
          <circle
            :r="nodeRadius(obj)"
            :fill="nodeColor(obj.labelName)"
            :stroke="isHazardWorker(obj) ? '#ef4444' : 'transparent'"
            :stroke-width="isHazardWorker(obj) ? 3 : 0"
            :opacity="hoveredNode && hoveredNode.id !== obj.id ? 0.35 : 1"
            class="node-circle"
          />
          <text
            :y="nodeRadius(obj) + 14"
            text-anchor="middle"
            font-size="11"
            fill="#475569"
            font-weight="500"
            :opacity="hoveredNode && hoveredNode.id !== obj.id ? 0.3 : 1"
          >{{ obj.labelName }}</text>
          <text
            v-if="isHazardWorker(obj)"
            :y="nodeRadius(obj) + 26"
            text-anchor="middle"
            font-size="9"
            fill="#ef4444"
            font-weight="600"
          >⚠</text>
        </g>
      </svg>

      <!-- Tooltip -->
      <div
        v-if="hoveredNode"
        class="node-tooltip"
        :style="{ left: tooltipPos.x + 'px', top: tooltipPos.y + 'px' }"
      >
        <div class="tt-label">{{ hoveredNode.labelName }}</div>
        <div class="tt-meta">置信度 {{ (hoveredNode.score * 100).toFixed(1) }}%</div>
        <div class="tt-meta">#{{ hoveredNode.objectIndex }}</div>
        <div v-if="findHazard(hoveredNode)" class="tt-hazard">
          <div class="tt-hazard-cat">{{ findHazard(hoveredNode).unsafeBehaviorCategory }}</div>
          <div class="tt-hazard-score">风险分 {{ (findHazard(hoveredNode).predScore * 100).toFixed(1) }}%</div>
        </div>
      </div>

      <!-- Legend -->
      <div class="graph-legend">
        <div class="legend-item"><span class="dot" style="background:#ff6b35"></span> 工人</div>
        <div class="legend-item"><span class="dot" style="background:#22c55e"></span> 安全设施/PPE</div>
        <div class="legend-item"><span class="dot" style="background:#ef4444"></span> 危险结构</div>
        <div class="legend-item"><span class="dot" style="background:#4a90d9"></span> 机械/设备</div>
        <div class="legend-item"><span class="dot" style="background:#f59e0b"></span> 材料</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { PictureFilled } from '@element-plus/icons-vue'

const props = defineProps({
  objects: { type: Array, default: () => [] },
  relations: { type: Array, default: () => [] },
  hazards: { type: Array, default: () => [] },
  imageWidth: { type: Number, default: 800 },
  imageHeight: { type: Number, default: 600 },
  imageUrl: { type: String, default: '' },
})

const svgRef = ref(null)
const hoveredNode = ref(null)
const tooltipPos = ref({ x: 0, y: 0 })

const svgW = ref(800)
const svgH = ref(500)
const padding = 60

// Position nodes using force-like layout
const positionedNodes = computed(() => {
  const objs = props.objects
  if (!objs.length) return []

  svgW.value = Math.max(800, props.imageWidth || 800)
  svgH.value = Math.max(500, props.imageHeight || 500)

  // Map image-space bbox center to SVG coordinates
  const imgW = props.imageWidth || 1
  const imgH = props.imageHeight || 1
  const sx = (svgW.value - padding * 2) / imgW
  const sy = (svgH.value - padding * 2) / imgH
  const scale = Math.min(sx, sy)
  const offsetX = (svgW.value - imgW * scale) / 2
  const offsetY = (svgH.value - imgH * scale) / 2

  return objs.map((obj, i) => {
    const bbox = obj.bbox || [0, 0, 100, 100]
    let cx, cy
    if (bbox[0] === 0 && bbox[1] === 0 && bbox[2] === 0 && bbox[3] === 0) {
      // No bbox: grid layout
      const cols = Math.ceil(Math.sqrt(objs.length))
      cx = padding + (i % cols) * ((svgW.value - padding * 2) / Math.max(cols - 1, 1))
      cy = padding + Math.floor(i / cols) * ((svgH.value - padding * 2) / Math.max(Math.ceil(objs.length / cols) - 1, 1))
    } else {
      cx = offsetX + ((bbox[0] + bbox[2]) / 2) * scale
      cy = offsetY + ((bbox[1] + bbox[3]) / 2) * scale
    }
    return { ...obj, x: cx, y: cy }
  })
})

const positionedRels = computed(() => {
  const nodeMap = {}
  positionedNodes.value.forEach(n => { nodeMap[n.id] = n })

  return props.relations.map(rel => {
    const subj = nodeMap[rel.subjectObjectId]
    const obj = nodeMap[rel.objectObjectId]
    if (!subj || !obj) return null
    const isHazard = props.hazards.some(
      h => h.workerObjectId === rel.subjectObjectId || h.workerObjectId === rel.objectObjectId
    )
    return {
      ...rel,
      x1: subj.x, y1: subj.y,
      x2: obj.x, y2: obj.y,
      isHazard,
      labelVisible: rel.score > 0.5,
    }
  }).filter(Boolean)
})

function nodeRadius(obj) {
  return obj.labelName && /worker|person|工人|人员/i.test(obj.labelName) ? 18 : 12
}

function nodeColor(label) {
  if (!label) return '#94a3b8'
  if (/worker|person|工人|人员/i.test(label)) return '#ff6b35'
  if (/helmet|vest|harness|belt|glove|goggle|安全帽|安全带|反光衣|手套|护目|ppe/i.test(label)) return '#22c55e'
  if (/scaffold|guardrail|edge|ladder|hole|opening|脚手架|栏杆|边缘|洞口/i.test(label)) return '#ef4444'
  if (/crane|excavator|machinery|vehicle|truck|吊车|挖掘机|机械|车辆/i.test(label)) return '#4a90d9'
  if (/material|brick|steel|concrete|pipe|材料|砖|钢筋/i.test(label)) return '#f59e0b'
  return '#94a3b8'
}

function isHazardWorker(obj) {
  return props.hazards.some(h => h.workerObjectId === obj.id)
}

function findHazard(obj) {
  return props.hazards.find(h => h.workerObjectId === obj.id) || null
}

function onMouseMove(e) {
  tooltipPos.value = { x: e.offsetX + 12, y: e.offsetY - 10 }
}
</script>

<style scoped>
.scene-graph-view {
  position: relative;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 60px 0;
  color: #94a3b8;
}

.graph-layout {
  position: relative;
}

.graph-canvas {
  width: 100%;
  height: auto;
  background: #fafbfc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  cursor: crosshair;
}

.graph-node {
  cursor: pointer;
  transition: opacity 0.2s;
}

.node-circle {
  transition: r 0.2s, opacity 0.2s;
}

.node-tooltip {
  position: absolute;
  pointer-events: none;
  background: #1e293b;
  color: #fff;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  z-index: 10;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  min-width: 120px;
}

.tt-label {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 2px;
}

.tt-meta {
  font-size: 11px;
  color: #94a3b8;
  font-family: 'JetBrains Mono', monospace;
}

.tt-hazard {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid rgba(255,255,255,0.15);
}

.tt-hazard-cat {
  font-weight: 600;
  color: #ef4444;
  font-size: 12px;
}

.tt-hazard-score {
  font-size: 11px;
  color: #f87171;
  font-family: 'JetBrains Mono', monospace;
}

.graph-legend {
  display: flex;
  gap: 20px;
  padding: 10px 16px;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #64748b;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}
</style>
