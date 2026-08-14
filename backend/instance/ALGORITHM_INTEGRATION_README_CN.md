# 算法集成说明

当前 demo 后端已经把三段算法源码和运行资源都收进项目内部：

- `backend/algorithms/detector_service`
- `backend/algorithms/scene_graph_service`
- `backend/algorithms/hazard_reasoning_service`
- `backend/resources/scene_graph`
- `backend/resources/hazard_reasoning`
- `backend/resources/hf_models`

其中：

- STW-YOLO 的自定义 `ultralytics` 已复制到 `backend/algorithms/detector_service/vendor/stw_yolo`
- 场景图 `predcls` 权重、字典、`h5`、`image_data.json`、`glove` 已复制到 `backend/resources/scene_graph`
- 双编码器危险推理权重、规则原型、上下文策略已复制到 `backend/resources/hazard_reasoning`
- `bert-base-uncased` 本地模型已复制到 `backend/resources/hf_models/bert-base-uncased`

## 统一配置文件

后续部署时，优先只修改：

- `backend/instance/algorithm_runtime.json`

这一份文件集中管理：

- detector 两套权重路径
- scene graph 推理权重和元数据路径
- hazard reasoning 权重、规则、BERT 本地模型路径
- 三个服务统一的运行设备 `cpu/cuda`

## 当前默认约定

- detector 权重仍放在项目根目录：`weight/stw` 和 `weight/yolo`
- scene graph 服务模式默认把 `img_dir` 指到 `backend/uploads`
- 这意味着正式接口推理时，上传图片后可以直接走服务，不再依赖外部 `VG_100K` 图片目录

## 关于全量评估

如果后续要跑 HTCL 的全量数据集评估，而不是单张服务推理，需要额外把：

- `scene_graph.runtime_config.paths.img_dir`

临时改成真实的 `VG_100K` 图片目录。服务推理本身不需要这一步。

## 当前状态

当前 demo 侧已经具备：

- 三段算法源码在 demo 内部可被后端直接调用
- `/api/infer/pipeline`、`/api/infer/scene-graph`、`/api/infer/reasoning` 的后端接入口
- 图像、检测结果、场景图关系、危险推理结果的数据库落表与接口
- 所有关键权重和规则路径集中到一份运行时配置文件

## 部署建议

部署时优先保证以下目录一起带走：

- `backend/algorithms`
- `backend/resources`
- `backend/instance/algorithm_runtime.json`
- 项目根目录 `weight`

这样后续无论是 Docker、远端 GPU 机，还是 Nginx 代理，只需要针对这一份配置文件做最小量改动。
