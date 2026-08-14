import request from './request'

/** GET /api/infer/task/list */
export function getInferenceTaskList(params) {
  return request.get('/infer/task/list', { params })
}

/** GET /api/infer/task/:id */
export function getInferenceTask(id) {
  return request.get(`/infer/task/${id}`)
}

/** POST /api/infer/pipeline */
export function createPipelineTask(data) {
  return request.post('/infer/pipeline', data)
}

/** POST /api/infer/pipeline/run */
export function runPipelineSync(data) {
  return request.post('/infer/pipeline/run', data)
}

/** POST /api/infer/scene-graph */
export function createSceneGraphTask(data) {
  return request.post('/infer/scene-graph', data)
}

/** POST /api/infer/reasoning */
export function createReasoningTask(data) {
  return request.post('/infer/reasoning', data)
}

/** POST /api/infer/task/:id/run */
export function runInferenceTask(id) {
  return request.post(`/infer/task/${id}/run`)
}

/** GET /api/infer/task/:id/objects */
export function getTaskObjects(id) {
  return request.get(`/infer/task/${id}/objects`)
}

/** GET /api/infer/task/:id/scene-graph */
export function getTaskSceneGraph(id) {
  return request.get(`/infer/task/${id}/scene-graph`)
}

/** GET /api/infer/task/:id/hazards */
export function getTaskHazards(id) {
  return request.get(`/infer/task/${id}/hazards`)
}

/** POST /api/infer/task/:id/stage/:stage/mock-save */
export function mockSaveStageOutput(taskId, stageName, data) {
  return request.post(`/infer/task/${taskId}/stage/${stageName}/mock-save`, data)
}
