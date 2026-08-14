import request from './request'

export function getReportList(params) {
  return request.get('/report/list', { params })
}

export function generateReport(data) {
  return request.post('/report/generate', data)
}

export function getReport(id) {
  return request.get(`/report/${id}`)
}

export function deleteReport(id) {
  return request.delete(`/report/${id}`)
}
