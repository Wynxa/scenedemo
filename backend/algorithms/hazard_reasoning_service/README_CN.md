# hazard_reasoning_service

这个目录是后段“不安全行为推理”模块的独立服务骨架，职责是把场景图转成双编码器推理输入，再输出每个工人的不安全行为预测结果。

## 当前定位

- 输入：一张图像对应的场景图 JSON
- 输出：`worker_results + image_level_result`
- 不依赖 `projects/hazard_reasoning/src` 运行
- 不处理目标检测和关系预测，只消费场景图服务的输出

## 必要文件

- `checkpoint`
  - 推荐：`E:/MasterDegree/Experiment/HTCL/projects/hazard_reasoning/results/exp2/best_model.pt`
- `prototype_inventory`
  - 推荐：`E:/MasterDegree/Experiment/HTCL/projects/hazard_reasoning/config/rule_prototypes_v1.json`
- `context_policy`
  - 推荐：`E:/MasterDegree/Experiment/HTCL/projects/hazard_reasoning/config/hazard_context_policies_v4.json`
- `model_cache_dir`
  - 本地 HuggingFace 模型缓存目录，里面至少要有 `bert-base-uncased`

## 运行配置

参考 [runtime_config_example.json](/E:/MasterDegree/Experiment/HTCL/services/hazard_reasoning_service/runtime_config_example.json)：

- `paths.checkpoint`：双编码器权重
- `paths.prototype_inventory`：原型库
- `paths.context_policy`：上下文策略
- `paths.model_cache_dir`：BERT 缓存目录
- `model.scene_text_field`：建议先用 `scene_text_structured`
- `device`：`cpu` 或 `cuda`

## 测试命令

如果手里是 Label Studio 的导出标注，例如 `extracted_annotation_004910.json`，先转场景图：

```powershell
python -m services.hazard_reasoning_service.app.scripts.convert_labelstudio_annotation_to_scene_graph `
  --input-json E:/MasterDegree/Experiment/HTCL/extracted_annotation_004910.json `
  --output-json E:/MasterDegree/Experiment/HTCL/tmp/extracted_annotation_004910_scene_graph.json `
  --image-id 004910
```

再执行危险推理：

```powershell
D:/anaconda/envs/cssmx/python.exe -m services.hazard_reasoning_service.app.scripts.run_hazard_reasoning_infer `
  --runtime-config E:/MasterDegree/Experiment/HTCL/tmp/hazard_reasoning_runtime_config_local.json `
  --scene-graph-json E:/MasterDegree/Experiment/HTCL/tmp/extracted_annotation_004910_scene_graph.json `
  --output-json E:/MasterDegree/Experiment/HTCL/tmp/extracted_annotation_004910_hazard_result.json
```

本地 `004910` 样例已经验证通过，当前输出结论为：

- `has_unsafe_behavior = true`
- `unsafe_worker_count = 1`
- `pred_major_label = general_work_unsafe`

如果你已经拿到的是标准场景图 JSON，则直接运行：

```powershell
python -m services.hazard_reasoning_service.app.scripts.run_hazard_reasoning_infer `
  --runtime-config services/hazard_reasoning_service/runtime_config_example.json `
  --scene-graph-json E:/MasterDegree/Experiment/HTCL/tmp/scene_graph_service_request_test_000002_out_standalone_check.json `
  --output-json E:/MasterDegree/Experiment/HTCL/tmp/hazard_reasoning_result.json
```

如果场景图里没有 `worker` 节点，输出会是：

- `worker_results = []`
- `image_level_result.has_unsafe_behavior = false`

## 与前两个模块的衔接

- 目标检测服务输出：图像、框、类别
- 场景图服务输出：`objects + relationships`
- 本服务输入：直接吃场景图服务输出 JSON

因此前端正式业务页仍然只需要调主入口 `pipeline`，但后端内部可以保持：

1. 检测服务
2. 场景图服务
3. 危险行为推理服务

## 输出结构

输出 JSON 主体：

- `scene_graph`
- `worker_results`
- `image_level_result`
- `meta`

其中：

- `worker_results` 是前端按工人绘制危险标签的直接数据源
- `image_level_result` 是整图级别结论
- `scene_graph` 可以直接复用前端场景图绘制

## 备注

- 离线环境下，`pretrained_model` 可以直接写成本地 HuggingFace 模型目录
- 如果本地只有 `config/tokenizer`，没有单独的 `pytorch_model.bin`，当前服务也能先按配置构建 BERT 骨架，再由你的 `checkpoint` 回填参数
