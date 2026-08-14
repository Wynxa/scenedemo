# 前后端联调接口清单

本文档对应当前 demo 后端实际接口，前端正式业务页优先围绕 `pipeline` 主流程联调。

## 推荐联调主线

前端正式页建议按这个顺序调用：

1. `POST /api/image/upload`
2. `POST /api/infer/pipeline/run`
3. `GET /api/infer/task/{taskId}`
4. `GET /api/infer/task/{taskId}/scene-graph`
5. `GET /api/infer/task/{taskId}/hazards`

这样可以兼顾：

- 图片上传与持久化
- 全流程推理执行
- 任务详情回显
- 场景图绘制
- 不安全行为结果展示

## 1. 上传图片

接口：`POST /api/image/upload`

请求方式：`form-data`

字段：

- `file`: 图片文件，必填
- `datasetId`: 可选
- `sourceType`: 可选，默认 `upload`

成功返回核心字段：

- `id`: 图片主键，后续推理直接传这个
- `fileUrl`: 前端图片访问路径
- `width`
- `height`
- `fileName`

## 2. 执行全流程推理

接口：`POST /api/infer/pipeline/run`

请求体：

```json
{
  "imageId": 1
}
```

成功返回核心结构：

```json
{
  "task": {},
  "stageRuns": [],
  "result": {
    "detector_result": {},
    "scene_graph_result": {},
    "hazard_result": {}
  }
}
```

前端重点关心：

- `task.id`
- `result.detector_result.objects`
- `result.scene_graph_result.relationships`
- `result.hazard_result.unsafe_behavior_results`
- `result.hazard_result.image_level_result`

## 3. 查询任务详情

接口：`GET /api/infer/task/{taskId}`

用途：

- 页面刷新后重新拉取任务主信息
- 查看任务状态、阶段状态、摘要结果

返回核心字段：

- `task`
- `image`
- `stageRuns`
- `summary`

## 4. 查询场景图结果

接口：`GET /api/infer/task/{taskId}/scene-graph`

返回结构：

```json
{
  "objects": [],
  "relationships": []
}
```

### objects 数组字段

- `id`
- `taskId`
- `stageRunId`
- `imageId`
- `objectIndex`
- `labelId`
- `labelName`
- `bbox`: `[x1, y1, x2, y2]`
- `score`
- `sourceModel`
- `sourceBranch`

前端绘制检测框时，主要使用：

- `objectIndex`
- `labelName`
- `bbox`
- `score`

### relationships 数组字段

- `id`
- `taskId`
- `stageRunId`
- `imageId`
- `subjectObjectId`
- `objectObjectId`
- `predicateId`
- `predicateName`
- `score`
- `allScoresJson`

前端绘制场景图时，建议：

- 节点来源于 `objects`
- 边来源于 `relationships`
- 文本显示使用 `predicateName`

注意：

- `subjectObjectId` 和 `objectObjectId` 是数据库对象主键，不是 `objectIndex`
- 前端绘制关系边时，需要先把 `objects[].id` 映射成对应节点

## 5. 查询危险推理结果

接口：`GET /api/infer/task/{taskId}/hazards`

返回结构：

```json
{
  "workerResults": [],
  "imageLevelResult": {}
}
```

### workerResults 字段

每个工人结果重点看：

- `workerObjectId`
- `contextId`
- `contextText`
- `relationType`
- `unsafeBehaviorCategory`
- `predMajorLabel`
- `predMajorLabelId`
- `predScore`
- `requiredPpeJson`
- `presentPpeJson`
- `missingPpeJson`
- `sceneText`
- `sceneTextStructured`
- `topkJson`

前端重点展示建议：

- 工人对应框
- 不安全行为类别 `unsafeBehaviorCategory`
- 置信度 `predScore`
- 缺失 PPE `missingPpeJson`

### imageLevelResult 字段

重点字段：

- `hasUnsafeBehavior`
- `unsafeWorkerCount`
- `highestRiskCategory`
- `highestRiskScore`
- `unsafeBehaviorResultsJson`

其中 `unsafeBehaviorResultsJson` 适合直接做右侧汇总面板。

## 推荐页面展示映射

一个完整业务页可以拆成四块：

- 原图与检测框层：显示 `objects`
- 场景图关系层：显示 `relationships`
- 工人级危险卡片：显示 `workerResults`
- 图像级汇总面板：显示 `imageLevelResult`

## 登录说明

这些接口当前都带 `@login_required`，前端联调前要先拿到登录态。

## 规则管理接口

如果后面前端需要做规则配置页，可以直接接：

- `GET /api/rule/list`
- `POST /api/rule`
- `PUT /api/rule/{id}`
- `DELETE /api/rule/{id}`
- `POST /api/rule/import-json`
- `GET /api/rule/export-json`

当前后端启动时会自动把内置 `hazard_context_policies_v4.json` 同步到 `hazard_rule` 表，规则页第一次打开就应该有默认数据。
