# scene_graph_service 独立评估重构方案

## 1. 当前目标

这一阶段的目标不是“兼容旧主线评估”，而是：

- 让 `scene_graph_service` 自己完成关系推理评估闭环
- 可以复用 HTCL 主逻辑代码
- 不能再运行时依赖 `maskrcnn_benchmark`
- 不能再依赖 `yacs/cfg`
- 不能再依赖主线 `VGDataset` 和 `vg_eval`

最终要验证的是：

- 新服务层在重写底层依赖后
- 对同一份 `h5/json/image` 数据
- 能否复现原始 HTCL 主线的推理和评估结果

## 2. 为什么 `legacy_compat` 不再是主入口

当前已有脚本：

- [run_dataset_eval_legacy_compat.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_dataset_eval_legacy_compat.py)

它的用途是：

- 复用主线 `h5`
- 复用主线 `vg_eval`
- 复用主线模型构建
- 只替换服务层输入预处理

所以它验证的是：

- “服务层预处理替换后是否漂移”

它不验证：

- “服务层是否已经独立”

因此，从现在开始：

- `legacy_compat` 只保留为历史过渡脚本
- 不再作为最终服务层测试入口

## 3. 独立评估闭环应该是什么

目标闭环应当变成：

`h5/json/image -> service dataset reader -> service model runner -> service postprocess -> service evaluator`

也就是说，最终需要在 `scene_graph_service` 内部拥有下面四层：

1. 数据读取层
   - 读取 `VG-SGG-with-attri.h5`
   - 读取 `image_data.json`
   - 读取 `VG-SGG-dicts-with-attri.json`
   - 输出服务层自己的样本结构

2. 推理执行层
   - 使用服务层自己的 `BoxListLite`
   - 使用服务层自己的 transforms / ops
   - 复用 HTCL 关系推理主逻辑
   - 不再 import `maskrcnn_benchmark`

3. 后处理层
   - 对象结果整理
   - 关系结果整理
   - 场景图构建

4. 评估层
   - 复现 `R@20/50/100`
   - 复现 `mR@20/50/100`
   - 复现 `ng-R`
   - 复现 `ng-mR`
   - 复现 `zR/ng-zR`
   - 复现 `A@20/50/100`

## 4. 当前剩余旧依赖边界

目前 `scene_graph_service` 里真正还直接依赖旧底层的文件主要有：

- [htcl_runtime_adapter.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/adapters/htcl_runtime_adapter.py)
- [legacy_boxlist_bridge.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/adapters/legacy_boxlist_bridge.py)
- [run_dataset_eval_legacy_compat.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_dataset_eval_legacy_compat.py)

其中依赖点主要是：

- `maskrcnn_benchmark.config.cfg`
- `build_detection_model`
- `DetectronCheckpointer`
- `maskrcnn_benchmark.structures.bounding_box.BoxList`
- 主线 `VGDataset`
- 主线 `vg_eval`

## 5. 接下来要替掉的模块

### 5.1 配置层

替换目标：

- 不再使用 `yacs`
- 不再依赖主线 `cfg`

服务层需要自己的配置对象，最少覆盖：

- 模型权重路径
- 设备
- 图像 resize / normalize 参数
- label / predicate 字典路径
- dataset 路径
- 评估 split

### 5.2 数据集层

替换目标：

- 不再使用主线 `VGDataset`

服务层需要自己的：

- `VGH5DatasetLite`
- split 选择
- `h5` box / label / relation 解析
- image filename / size 对齐

### 5.3 结构层

替换目标：

- 不再把 `BoxListLite` 转回 legacy `BoxList`

服务层自己的 proposal / target / result 结构要继续扩展，直到能直接喂给新迁出的关系推理逻辑。

### 5.4 推理层

替换目标：

- 不再通过 `build_detection_model`
- 不再通过 `DetectronCheckpointer`

这一层应当把 HTCL 关系推理最小运行图迁入服务目录，只保留必要组件。

### 5.5 评估层

替换目标：

- 不再调用主线 `evaluate/vg_eval`

服务层要有自己的 evaluator，实现与主线同口径的指标统计。

## 6. 推荐分阶段落地顺序

### 阶段 A：数据与配置独立

完成：

- 服务层配置对象
- 服务层 `VGH5DatasetLite`
- 服务层样本读取脚本

验证目标：

- 不 import `maskrcnn_benchmark`
- 能正确读出图像、框、标签、关系

### 阶段 B：推理执行独立

完成：

- 去掉 `legacy_boxlist_bridge`
- 去掉 `htcl_runtime_adapter` 对旧 cfg / model builder 的依赖
- 建立服务层自己的模型 runner

验证目标：

- 单图推理跑通
- 输出对象、关系、场景图

### 阶段 C：评估独立

完成：

- 服务层 evaluator
- 新的服务层评估脚本

验证目标：

- 对同一份数据和权重
- 输出与主线接近的完整关系评估指标

## 7. 新测试入口原则

后续的主测试入口应当是新的服务层评估脚本，而不是 `legacy_compat`。

建议命名方向：

- `run_dataset_eval_service_owned.py`

它必须满足：

- 顶层不 import `maskrcnn_benchmark`
- 顶层不 import `yacs`
- 仅依赖 `scene_graph_service` 自己的模块

## 8. 当前结论

从这一阶段开始，我们对 `scene_graph_service` 的目标定义为：

- 复用 HTCL 主逻辑
- 但不再复用旧底层运行时

因此，后续所有新增代码都应优先服务于“独立评估闭环”，而不是继续扩展 `legacy_compat` 过渡方案。
