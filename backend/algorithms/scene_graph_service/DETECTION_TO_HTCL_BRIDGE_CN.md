# 检测服务到 HTCL 服务的衔接说明

## 当前结论

这一阶段可以把三个模块之间的职责划分成下面这样：

1. 第一个服务负责目标检测。
2. 第二个服务负责关系推理，也就是当前 `scene_graph_service`。
3. 第三个服务负责后续双编码器危险行为推理。

对第二个服务来说，来自第一个服务的输入不需要是裁剪图，也不需要是视觉特征，只需要：

- 原图路径
- 检测框
- 检测类别
- 可选检测分数

也就是说，`HTCL` 关系服务内部仍然自己做：

- 原图读取
- `backbone + FPN` 特征提取
- ROI 特征抽取
- union feature 抽取
- 关系预测

第一个检测服务只负责把检测结果整理成约定格式传进来。

## 为什么这样拆最稳

这样拆的好处是：

- 检测服务和关系服务边界清楚
- 后续更换检测模型时，不需要改 HTCL 关系服务内部逻辑
- 不需要在两个服务之间传大尺寸中间特征
- 前端需要展示检测框时，也能直接复用检测服务的结果

这里最关键的点是：

- 关系服务吃的是“原图 + 框 + 标签”
- 不是“裁切后的 worker 图块”
- 也不是“检测服务抽出来的中间视觉特征”

## 推荐的请求协议

HTCL 服务建议统一接收下面这类 JSON：

```json
{
  "image_id": "test_000002",
  "image_path": "E:/MasterDegree/Experiment/HTCL/datasets/construction_vg_raw_train_test_full_20260703/vg/VG_100K/test_000002.jpg",
  "detector_service": "stw_yolo_service",
  "detector_model": "stw_yolo_dual_branch",
  "objects": [
    {
      "object_id": 0,
      "label_id": 1,
      "label": "worker",
      "bbox": [100, 120, 260, 580],
      "score": 0.98
    }
  ]
}
```

其中 `bbox` 统一使用 `xyxy`。

如果上游检测服务输出不是这个字段名，也没关系，当前已经做了一个适配层，支持这些别名：

- `objects` / `detections`
- `bbox` / `bbox_xyxy` / `box`
- `label_id` / `class_id` / `category_id`
- `label` / `name` / `class_name` / `category_name`
- `score` / `conf` / `confidence`
- 或者 `x,y,w,h`

## 已落地的适配代码

适配器：

- [detector_payload_adapter.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/services/detector_payload_adapter.py)

调试脚本：

- [run_scene_graph_from_detector_payload.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_scene_graph_from_detector_payload.py)

这个脚本的作用是：

- 读取检测服务风格的输入 JSON
- 归一化成 HTCL 服务需要的 `objects`
- 调用当前关系推理入口
- 输出场景图结果

## 当前工程上的建议

目前最合适的落地路线是：

1. 第一个检测服务先稳定输出统一 JSON。
2. `scene_graph_service` 先稳定吃这份 JSON。
3. `backbone + FPN` 这一段在 HTCL 服务内部保留为视觉子模块。
4. 等关系服务完全稳定之后，再考虑是否继续把视觉层也彻底去耦。

## 下一步开发重点

下一步不是改协议，而是补齐 `scene_graph_service` 里的视觉特征入口：

- 用 HTCL 主线视觉部分提供真实 FPN 特征图
- 再接我们已经迁好的 `RelationHeadLite`
- 最后替换 [htcl_service_runner.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/runner/htcl_service_runner.py) 里的 `NotImplementedError`

这样做，整个第二个服务就能真正形成：

- 外部输入：原图 + 检测框 + 标签
- 内部处理：视觉特征 + 关系预测
- 外部输出：对象 + 关系 + 场景图
