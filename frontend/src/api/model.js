import request from './request'

export function getModelList(params) {
  return request.get('/model/list', { params })
}

export function registerModel(data) {
  return request.post('/model/register', data)
}

export function getModel(id) {
  return request.get(`/model/${id}`)
}

export function updateModel(id, data) {
  return request.put(`/model/${id}`, data)
}

export function activateModel(id) {
  return request.post(`/model/${id}/activate`)
}

export function deleteModel(id) {
  return request.delete(`/model/${id}`)
}

export function createTraining(data) {
  return request.post('/model/training', data)
}

export function getTrainingList(params) {
  return request.get('/model/training/list', { params })
}

export function getTraining(id) {
  return request.get(`/model/training/${id}`)
}
