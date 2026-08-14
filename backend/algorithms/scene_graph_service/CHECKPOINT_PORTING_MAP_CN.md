# scene_graph_service 权重迁移映射

## 1. 目的

这个文档用于把 HTCL 原始 checkpoint 的权重键，映射到 `scene_graph_service` 后续要建立的服务层模块。

目标不是立即完成所有模块重写，而是先把迁移边界固定下来，避免后续前向迁移时结构混乱。

## 2. 当前 checkpoint 结构概览

以当前测试权重：

- `E:\MasterDegree\Experiment\HTCL\results\precls\ours\model_0006000.pth`

服务层独立检查结果显示：

- `num_keys = 721`
- 顶层前缀主要是：
  - `backbone`
  - `roi_heads`
  - `rpn`

这说明当前 checkpoint 是完整检测 + 关系推理模型，不是只有关系头的小权重。

## 3. 服务层模块映射

### 3.1 `backbone`

匹配前缀：

- `backbone.`

职责：

- 主干网络
- FPN 特征层

迁移建议：

- 后续建立 `service_backbone` 模块
- 优先保证张量输出层级与旧主线一致

### 3.2 `rpn`

匹配前缀：

- `rpn.`

职责：

- proposal 生成

迁移建议：

- 如果服务最终长期只做 `predcls/precls`，这一层可以延后
- 因为当前场景下输入已有 bbox，真实重构优先级低于 relation head

### 3.3 `roi_box_head`

匹配前缀：

- `roi_heads.box.`

职责：

- ROI box 特征提取
- box predictor

迁移建议：

- 如果关系推理仍然需要 ROI 级视觉特征，这一层必须迁
- 但若当前阶段只做 `predcls` 且可直接用 GT box / label，可以先围绕关系头的最低必需链路迁移

### 3.4 `roi_relation_head`

匹配前缀：

- `roi_heads.relation.`

职责：

- relation feature extractor
- context encoder
- relation predictor

迁移建议：

- 这是服务层迁移的核心模块
- 后续 `HTCLServiceRunner._predict_prepared(...)` 应先围绕这一层补齐

### 3.5 `roi_attribute_head`

匹配前缀：

- `roi_heads.attribute.`

职责：

- attribute head

迁移建议：

- 当前施工场景图主任务不是 attribute，优先级最低

## 4. 当前迁移优先级

建议按下面顺序推进：

1. `roi_relation_head`
2. `roi_box_head` 中关系推理所必需的最小视觉特征链
3. `backbone`
4. `rpn`
5. `roi_attribute_head`

注意：

- 权重里虽然包含完整模型
- 但服务层重构不一定要按原模块顺序 1:1 搬迁
- 应优先按“当前推理模式最小可运行链路”迁移

## 5. 当前结论

对 `scene_graph_service` 来说，真实迁移的关键不是“先把所有模块都重写”，而是：

- 先把 `checkpoint -> service module` 的映射固定
- 再以 `roi_relation_head` 为中心，逐步把服务层真实 runner 接起来

因此，这份映射文档将作为后续 `HTCLServiceRunner` 前向迁移的骨架参考。
