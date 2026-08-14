import request from './request'

export function getRuleList(params) {
  return request.get('/rule/list', { params })
}

export function createRule(data) {
  return request.post('/rule', data)
}

export function getRule(id) {
  return request.get(`/rule/${id}`)
}

export function updateRule(id, data) {
  return request.put(`/rule/${id}`, data)
}

export function deleteRule(id) {
  return request.delete(`/rule/${id}`)
}

export function importRules(data) {
  return request.post('/rule/import-json', data)
}

export function exportRules() {
  return request.get('/rule/export-json')
}
