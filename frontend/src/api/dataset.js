import request from './request'

export function getDatasetList(params) {
  return request.get('/dataset/list', { params })
}

export function createDataset(data) {
  return request.post('/dataset', data)
}

export function getDataset(id) {
  return request.get(`/dataset/${id}`)
}

export function deleteDataset(id) {
  return request.delete(`/dataset/${id}`)
}

export function uploadImages(datasetId, formData) {
  return request.post(`/dataset/${datasetId}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function getDatasetImages(datasetId, params) {
  return request.get(`/dataset/${datasetId}/images`, { params })
}
