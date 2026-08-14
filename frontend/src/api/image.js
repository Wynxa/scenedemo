import request from './request'

export function getImageList(params) {
  return request.get('/image/list', { params })
}

export function getImage(id) {
  return request.get(`/image/${id}`)
}

export function uploadImage(formData) {
  return request.post('/image/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
