<template>
  <div class="dataset-detail">
    <el-page-header @back="$router.back()">
      <template #content>
        <span>{{ dataset?.name }} — 图片管理</span>
      </template>
    </el-page-header>

    <el-card shadow="never" style="margin-top:12px">
      <!-- Upload Area -->
      <el-upload
        drag
        multiple
        :action="`/api/dataset/${route.params.id}/upload`"
        :headers="uploadHeaders"
        :on-success="onUploadSuccess"
        :before-upload="beforeUpload"
        accept=".jpg,.jpeg,.png,.bmp,.webp"
      >
        <el-icon :size="48"><UploadFilled /></el-icon>
        <div class="upload-text">拖拽图片到此处，或点击上传</div>
        <div class="upload-hint">支持 JPG / PNG / BMP / WEBP，单次最多50张</div>
      </el-upload>
    </el-card>

    <!-- Image Grid -->
    <el-card shadow="never" style="margin-top:12px">
      <div v-loading="loading" class="image-grid">
        <div v-for="img in images" :key="img.id" class="image-item" @click="previewImage = img">
          <el-image :src="`/uploads/${img.filePath}`" fit="cover" style="width:100%;height:160px" lazy>
            <template #error><div class="image-error">加载失败</div></template>
          </el-image>
          <div class="image-name">{{ img.fileName }}</div>
          <div class="image-meta">{{ img.width }}x{{ img.height }}</div>
        </div>
        <el-empty v-if="!loading && images.length === 0" description="暂无图片" />
      </div>
      <el-pagination
        style="margin-top:16px;justify-content:flex-end"
        v-model:current-page="pageNum"
        v-model:page-size="pageSize"
        :total="total"
        layout="total, prev, pager, next"
        @change="fetchImages"
      />
    </el-card>

    <!-- Preview Dialog -->
    <el-dialog v-model="showPreview" title="图片预览" width="80%">
      <el-image v-if="previewImage" :src="`/uploads/${previewImage.filePath}`" fit="contain"
        style="width:100%;max-height:70vh" />
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { getDataset, getDatasetImages } from '@/api/dataset'

const route = useRoute()
const dataset = ref(null)
const images = ref([])
const total = ref(0)
const pageNum = ref(1)
const pageSize = ref(20)
const loading = ref(false)
const previewImage = ref(null)
const showPreview = ref(false)

const uploadHeaders = {
  Authorization: `Bearer ${localStorage.getItem('token')}`,
}

function beforeUpload(file) {
  const valid = /\.(jpg|jpeg|png|bmp|webp)$/i.test(file.name)
  if (!valid) ElMessage.error('不支持的文件格式')
  return valid
}

function onUploadSuccess() {
  ElMessage.success('上传成功')
  fetchImages()
}

async function fetchImages() {
  loading.value = true
  try {
    const res = await getDatasetImages(route.params.id, { pageNum: pageNum.value, pageSize: pageSize.value })
    images.value = res.rows
    total.value = res.total
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  dataset.value = await getDataset(route.params.id)
  fetchImages()
})
</script>

<style scoped>
.dataset-detail { }
.upload-text { font-size: 15px; color: #666; margin-top: 8px; }
.upload-hint { font-size: 12px; color: #999; margin-top: 4px; }
.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
}
.image-item {
  border: 1px solid #eee;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: box-shadow 0.2s;
}
.image-item:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.12); }
.image-name { font-size: 13px; color: #333; padding: 6px 8px 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.image-meta { font-size: 12px; color: #999; padding: 0 8px 8px; }
.image-error { display: flex; align-items: center; justify-content: center; height: 160px; background: #f5f5f5; color: #999; font-size: 13px; }
</style>
