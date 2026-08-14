import request from './request'

export function createDetectionTask(data) {
  return request.post('/detection/task', data)
}

export function getDetectionTaskList(params) {
  return request.get('/detection/task/list', { params })
}

export function getDetectionTask(id) {
  return request.get(`/detection/task/${id}`)
}

export function deleteDetectionTask(id) {
  return request.delete(`/detection/task/${id}`)
}

export function getDetectionResults(taskId, params) {
  return request.get(`/detection/result/${taskId}`, { params })
}

export function getResultDetail(taskId, imageId) {
  return request.get(`/detection/result/${taskId}/${imageId}`)
}

export function singleDetection(formData) {
  return request.post('/detection/single', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
