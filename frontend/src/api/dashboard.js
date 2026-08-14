import request from './request'

export function getDashboardStats() {
  return request.get('/dashboard/stats')
}

export function getBehaviorDistribution() {
  return request.get('/dashboard/charts/behavior-dist')
}

export function getEntityDistribution() {
  return request.get('/dashboard/charts/entity-dist')
}
