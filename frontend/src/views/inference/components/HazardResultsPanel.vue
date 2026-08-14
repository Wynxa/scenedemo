<template>
  <div class="hazard-results-panel">
    <div v-if="!workerResults.length" class="empty-state">
      <el-icon :size="48"><WarningFilled /></el-icon>
      <p>运行危险推理后在此查看分析结果</p>
    </div>

    <div v-else class="hazard-grid">
      <el-card
        v-for="(w, idx) in workerResults"
        :key="w.id || idx"
        shadow="hover"
        class="hazard-card"
        :class="{ 'is-unsafe': w.unsafeBehaviorCategory && w.unsafeBehaviorCategory !== 'safe' }"
      >
        <div class="hc-header">
          <span class="hc-worker">
            <el-icon><UserFilled /></el-icon>
            Worker #{{ idx + 1 }}
          </span>
          <el-tag
            :type="w.unsafeBehaviorCategory && w.unsafeBehaviorCategory !== 'safe' ? 'danger' : 'success'"
            size="small"
            effect="dark"
          >
            {{ w.unsafeBehaviorCategory && w.unsafeBehaviorCategory !== 'safe'
              ? formatCategory(w.unsafeBehaviorCategory) : '安全' }}
          </el-tag>
        </div>

        <div class="hc-body">
          <div class="hc-field">
            <span class="field-label">场景上下文</span>
            <span class="field-value">{{ w.contextText || '-' }}</span>
          </div>
          <div class="hc-field">
            <span class="field-label">触发关系</span>
            <span class="field-value">{{ w.relationType || '-' }}</span>
          </div>
          <div class="hc-field">
            <span class="field-label">预测行为</span>
            <span class="field-value mono">{{ w.predMajorLabel || '-' }}</span>
          </div>
          <div class="hc-field">
            <span class="field-label">置信度</span>
            <span class="field-value mono">
              <el-progress
                :percentage="(w.predScore || 0) * 100"
                :stroke-width="6"
                :color="w.predScore > 0.5 ? '#ef4444' : '#22c55e'"
                style="width: 120px; display: inline-flex"
              />
              <span style="margin-left: 8px">{{ ((w.predScore || 0) * 100).toFixed(1) }}%</span>
            </span>
          </div>

          <!-- PPE Analysis -->
          <div v-if="w.missingPpeJson && w.missingPpeJson.length" class="ppe-section">
            <div class="ppe-title">缺少的 PPE</div>
            <div class="ppe-tags">
              <el-tag v-for="ppe in w.missingPpeJson" :key="ppe" size="small" type="danger" effect="plain">
                {{ ppe }}
              </el-tag>
            </div>
          </div>
          <div v-if="w.presentPpeJson && w.presentPpeJson.length" class="ppe-section">
            <div class="ppe-title">已穿戴的 PPE</div>
            <div class="ppe-tags">
              <el-tag v-for="ppe in w.presentPpeJson" :key="ppe" size="small" type="success" effect="plain">
                {{ ppe }}
              </el-tag>
            </div>
          </div>

          <!-- Scene description -->
          <div v-if="w.sceneText" class="hc-scene">
            {{ w.sceneText }}
          </div>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { WarningFilled, UserFilled } from '@element-plus/icons-vue'

defineProps({
  workerResults: { type: Array, default: () => [] },
})

const categoryMap = {
  climbing_scaffold_frame: '攀爬脚手架',
  leaning_out: '身体探出',
  standing_on_guardrail: '站立在护栏上',
  throwing_material: '抛掷物料',
  unsafe_posture: '不安全姿态',
  missing_step: '踏空/失足',
  no_helmet: '未戴安全帽',
  no_harness: '未系安全带',
  no_vest: '未穿反光衣',
}

function formatCategory(cat) {
  return categoryMap[cat] || cat
}
</script>

<style scoped>
.hazard-results-panel {
  min-height: 200px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 60px 0;
  color: #94a3b8;
}

.hazard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 16px;
}

.hazard-card {
  border: 1px solid #e2e8f0;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.hazard-card.is-unsafe {
  border-left: 3px solid #ef4444;
}

.hc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.hc-worker {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  font-size: 14px;
  color: #1e293b;
}

.hc-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.hc-field {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.field-label {
  font-size: 12px;
  color: #94a3b8;
}

.field-value {
  font-size: 13px;
  color: #334155;
  font-weight: 500;
  max-width: 60%;
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.field-value.mono {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
  font-size: 12px;
}

.ppe-section {
  margin-top: 4px;
  padding-top: 8px;
  border-top: 1px solid #f1f5f9;
}

.ppe-title {
  font-size: 11px;
  color: #94a3b8;
  margin-bottom: 4px;
}

.ppe-tags {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.hc-scene {
  margin-top: 8px;
  padding: 8px 10px;
  background: #f8fafc;
  border-radius: 4px;
  font-size: 12px;
  color: #64748b;
  line-height: 1.5;
}
</style>
