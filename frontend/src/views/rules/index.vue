<template>
  <div class="rules-page">
    <el-card shadow="never">
      <div class="toolbar">
        <div class="toolbar-left">
          <el-select v-model="filterType" placeholder="规则类型" clearable style="width:160px" @change="fetchList">
            <el-option label="上下文策略" value="context_policy" />
            <el-option label="空间约束" value="spatial_constraint" />
            <el-option label="PPE 合规" value="ppe_compliance" />
            <el-option label="行为规则" value="behavior_rule" />
          </el-select>
          <el-select v-model="filterStatus" placeholder="状态" clearable style="width:120px;margin-left:8px" @change="fetchList">
            <el-option label="启用" value="active" />
            <el-option label="禁用" value="inactive" />
          </el-select>
        </div>
        <div class="toolbar-right">
          <el-button @click="handleExport">导出 JSON</el-button>
          <el-upload
            :auto-upload="false"
            :show-file-list="false"
            accept=".json"
            :on-change="handleImportFile"
            style="display:inline-block;margin-left:8px"
          >
            <el-button>导入 JSON</el-button>
          </el-upload>
          <el-button type="primary" style="margin-left:8px" @click="openCreate">新建规则</el-button>
        </div>
      </div>
    </el-card>

    <el-card shadow="never" style="margin-top:12px">
      <el-table :data="list" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="ruleCode" label="规则编码" width="180">
          <template #default="{ row }">
            <code class="mono">{{ row.ruleCode }}</code>
          </template>
        </el-table-column>
        <el-table-column prop="ruleName" label="规则名称" min-width="160" />
        <el-table-column label="类型" width="120">
          <template #default="{ row }">
            <el-tag size="small" :type="typeTag(row.ruleType)">{{ typeLabel(row.ruleType) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="relationType" label="触发关系" width="120" />
        <el-table-column prop="priority" label="优先级" width="80" align="center">
          <template #default="{ row }">
            <span class="priority" :class="'p' + (row.priority >= 7 ? 'high' : row.priority >= 4 ? 'mid' : 'low')">
              {{ row.priority }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="80">
          <template #default="{ row }">
            <el-tag size="small" :type="row.status === 'active' ? 'success' : 'info'">
              {{ row.status === 'active' ? '启用' : '禁用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="version" label="版本" width="80" />
        <el-table-column label="操作" width="220" fixed="right">
          <template #default="{ row }">
            <el-button size="small" link type="primary" @click="openEdit(row)">编辑</el-button>
            <el-button size="small" link type="primary" @click="openDetail(row)">详情</el-button>
            <el-button size="small" link type="danger" @click="handleDelete(row)">删除</el-button>
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

    <!-- Create/Edit Dialog -->
    <el-dialog
      v-model="showDialog"
      :title="editingRule ? '编辑规则' : '新建规则'"
      width="640px"
      destroy-on-close
    >
      <el-form ref="formRef" :model="form" :rules="rules" label-width="110px" label-position="left">
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="规则编码" prop="ruleCode">
              <el-input v-model="form.ruleCode" placeholder="如 ctx_w03_02" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="规则名称" prop="ruleName">
              <el-input v-model="form.ruleName" placeholder="如 无安全带站立于边缘" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="规则类型">
              <el-select v-model="form.ruleType" style="width:100%">
                <el-option label="上下文策略" value="context_policy" />
                <el-option label="空间约束" value="spatial_constraint" />
                <el-option label="PPE 合规" value="ppe_compliance" />
                <el-option label="行为规则" value="behavior_rule" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="触发关系">
              <el-input v-model="form.relationType" placeholder="如 standing_on" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="上下文 ID">
              <el-input v-model="form.contextId" placeholder="如 w03" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="匹配模式">
              <el-select v-model="form.matchMode" style="width:100%">
                <el-option label="任一匹配" value="any" />
                <el-option label="全部匹配" value="all" />
                <el-option label="精确匹配" value="exact" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="优先级">
              <el-input-number v-model="form.priority" :min="0" :max="10" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="状态">
              <el-switch
                v-model="form.status"
                active-value="active"
                inactive-value="inactive"
                active-text="启用"
                inactive-text="禁用"
              />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="规则说明">
          <el-input v-model="form.ruleText" type="textarea" :rows="3" placeholder="描述此规则在什么条件下触发..." />
        </el-form-item>
        <el-form-item label="要求 PPE">
          <el-select
            v-model="form.requiredPpeJson"
            multiple
            filterable
            allow-create
            placeholder="输入 PPE 名称后回车添加"
            style="width:100%"
          >
            <el-option label="安全帽" value="安全帽" />
            <el-option label="安全带" value="安全带" />
            <el-option label="反光衣" value="反光衣" />
            <el-option label="防护手套" value="防护手套" />
            <el-option label="护目镜" value="护目镜" />
            <el-option label="安全鞋" value="安全鞋" />
          </el-select>
        </el-form-item>
        <el-form-item label="触发关系配置">
          <el-input
            v-model="triggerRelationsText"
            type="textarea"
            :rows="3"
            placeholder='JSON array, e.g. [{"subject":"worker","predicate":"standing_on","object":"guardrail"}]'
          />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="form.remark" placeholder="可选备注" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">取消</el-button>
        <el-button type="primary" @click="handleSave" :loading="saving">
          {{ editingRule ? '更新' : '创建' }}
        </el-button>
      </template>
    </el-dialog>

    <!-- Detail Dialog -->
    <el-dialog v-model="showDetail" title="规则详情" width="560px">
      <template v-if="detailRule">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="规则编码">{{ detailRule.ruleCode }}</el-descriptions-item>
          <el-descriptions-item label="规则名称">{{ detailRule.ruleName }}</el-descriptions-item>
          <el-descriptions-item label="类型">{{ typeLabel(detailRule.ruleType) }}</el-descriptions-item>
          <el-descriptions-item label="优先级">{{ detailRule.priority }}</el-descriptions-item>
          <el-descriptions-item label="上下文 ID">{{ detailRule.contextId || '-' }}</el-descriptions-item>
          <el-descriptions-item label="触发关系">{{ detailRule.relationType || '-' }}</el-descriptions-item>
          <el-descriptions-item label="匹配模式">{{ detailRule.matchMode }}</el-descriptions-item>
          <el-descriptions-item label="版本">{{ detailRule.version }}</el-descriptions-item>
          <el-descriptions-item label="规则说明" :span="2">{{ detailRule.ruleText || '-' }}</el-descriptions-item>
          <el-descriptions-item label="要求 PPE" :span="2">
            <el-tag v-for="ppe in (detailRule.requiredPpeJson || [])" :key="ppe" size="small" style="margin-right:4px">
              {{ ppe }}
            </el-tag>
            <span v-if="!detailRule.requiredPpeJson?.length">-</span>
          </el-descriptions-item>
          <el-descriptions-item label="触发关系配置" :span="2">
            <code class="mono">{{ JSON.stringify(detailRule.triggerRelationsJson || []) }}</code>
          </el-descriptions-item>
        </el-descriptions>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import dayjs from 'dayjs'
import { getRuleList, createRule, updateRule, deleteRule, exportRules } from '@/api/rule'
import request from '@/api/request'

const loading = ref(false)
const list = ref([])
const total = ref(0)
const pageNum = ref(1)
const pageSize = ref(10)
const filterType = ref('')
const filterStatus = ref('')

const showDialog = ref(false)
const showDetail = ref(false)
const editingRule = ref(null)
const detailRule = ref(null)
const saving = ref(false)
const formRef = ref(null)
const form = reactive({
  ruleCode: '', ruleName: '', ruleType: 'context_policy', contextId: '',
  relationType: '', matchMode: 'any', priority: 0, requiredPpeJson: [],
  triggerRelationsJson: [], ruleText: '', status: 'active', version: '1.0.0', remark: '',
})
const triggerRelationsText = ref('')
const rules = {
  ruleCode: [{ required: true, message: '请输入规则编码', trigger: 'blur' }],
  ruleName: [{ required: true, message: '请输入规则名称', trigger: 'blur' }],
}

function typeTag(t) {
  return { context_policy: '', spatial_constraint: 'warning', ppe_compliance: 'success', behavior_rule: 'danger' }[t] || 'info'
}
function typeLabel(t) {
  return { context_policy: '上下文策略', spatial_constraint: '空间约束', ppe_compliance: 'PPE 合规', behavior_rule: '行为规则' }[t] || t
}

async function fetchList() {
  loading.value = true
  try {
    const params = { pageNum: pageNum.value, pageSize: pageSize.value }
    if (filterType.value) params.ruleType = filterType.value
    if (filterStatus.value) params.status = filterStatus.value
    const res = await getRuleList(params)
    list.value = res.rows
    total.value = res.total
  } finally { loading.value = false }
}

function resetForm() {
  Object.assign(form, {
    ruleCode: '', ruleName: '', ruleType: 'context_policy', contextId: '',
    relationType: '', matchMode: 'any', priority: 0, requiredPpeJson: [],
    triggerRelationsJson: [], ruleText: '', status: 'active', version: '1.0.0', remark: '',
  })
  triggerRelationsText.value = ''
}

function openCreate() {
  editingRule.value = null
  resetForm()
  showDialog.value = true
}

function openEdit(row) {
  editingRule.value = row
  Object.assign(form, {
    ruleCode: row.ruleCode,
    ruleName: row.ruleName,
    ruleType: row.ruleType,
    contextId: row.contextId || '',
    relationType: row.relationType || '',
    matchMode: row.matchMode || 'any',
    priority: row.priority || 0,
    requiredPpeJson: [...(row.requiredPpeJson || [])],
    triggerRelationsJson: [...(row.triggerRelationsJson || [])],
    ruleText: row.ruleText || '',
    status: row.status || 'active',
    version: row.version || '1.0.0',
    remark: row.remark || '',
  })
  triggerRelationsText.value = JSON.stringify(row.triggerRelationsJson || [], null, 2)
  showDialog.value = true
}

async function handleSave() {
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return
  saving.value = true
  try {
    let parsedTriggers = []
    try { parsedTriggers = JSON.parse(triggerRelationsText.value || '[]') } catch { /* ok */ }
    const payload = {
      ...form,
      triggerRelationsJson: parsedTriggers,
    }
    if (editingRule.value) {
      await updateRule(editingRule.value.id, payload)
      ElMessage.success('规则更新成功')
    } else {
      await createRule(payload)
      ElMessage.success('规则创建成功')
    }
    showDialog.value = false
    fetchList()
  } finally { saving.value = false }
}

async function handleDelete(row) {
  await ElMessageBox.confirm(`确定删除规则 "${row.ruleName}"？`, '警告', { type: 'warning' })
  await deleteRule(row.id)
  ElMessage.success('删除成功')
  fetchList()
}

function openDetail(row) {
  detailRule.value = row
  showDetail.value = true
}

async function handleExport() {
  const data = await exportRules()
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = `rules_${dayjs().format('YYYYMMDD_HHmmss')}.json`; a.click()
  URL.revokeObjectURL(url)
  ElMessage.success('导出成功')
}

async function handleImportFile(file) {
  try {
    const text = await file.raw.text()
    const json = JSON.parse(text)
    await request.post('/rule/import-json', json)
    ElMessage.success('规则导入成功')
    fetchList()
  } catch (e) {
    ElMessage.error('导入失败: ' + (e.message || 'JSON 格式错误'))
  }
}

onMounted(fetchList)
</script>

<style scoped>
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.mono {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
  font-size: 12px;
  background: #f1f5f9;
  padding: 2px 6px;
  border-radius: 3px;
  color: #334155;
}

.priority {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  font-weight: 700;
  font-size: 13px;
}

.priority.p-high { background: #fef2f2; color: #ef4444; }
.priority.p-mid { background: #fff7ed; color: #f97316; }
.priority.p-low { background: #f8fafc; color: #94a3b8; }
</style>
