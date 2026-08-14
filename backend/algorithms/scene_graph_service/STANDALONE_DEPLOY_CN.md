# scene_graph_service 独立部署说明

## 当前结论

现在的 `scene_graph_service` 已经整理成“目录内自带运行源码”的形态。

也就是说，这个目录内部已经包含：

- 服务层自有代码
- 运行时需要的 `maskrcnn_benchmark` 源码副本
- 运行时需要的 `construction_predcls` 配置副本

当前目录下新增了：

- [vendor/maskrcnn_benchmark](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/vendor/maskrcnn_benchmark)
- [vendor/projects/construction_predcls](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/vendor/projects/construction_predcls)

## 这意味着什么

你把 `scene_graph_service` 这个目录单独拷到远端时：

- 不再需要依赖仓库根目录里的 `maskrcnn_benchmark`
- 不再需要依赖仓库根目录里的 `projects/construction_predcls`
- 推理主链路不再要求先编译 `maskrcnn_benchmark._C`

当前主链路优先从 `scene_graph_service/vendor` 里加载源码。

## 仍然需要你在远端准备的内容

这个目录虽然已经“源码自带”，但运行时仍然需要外部资源：

- 数据集文件
- 图像目录
- `VG-SGG-with-attri.h5`
- `VG-SGG-dicts-with-attri.json`
- `image_data.json`
- 关系权重 `model_0006000.pth`
- GloVe 文件目录

这些资源不放在服务目录内部，原因是它们体积大，而且你远端路径可能会变。

## 当前推荐入口

远端单图验证优先使用：

- [run_htcl_service_request_infer.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_htcl_service_request_infer.py)

检测服务风格输入验证使用：

- [run_scene_graph_from_detector_payload.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_scene_graph_from_detector_payload.py)

## 运行时配置

示例配置文件：

- [runtime_config_example.json](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/runtime_config_example.json)

这个配置里，内部源码相关路径已经改成了 `__SERVICE_ROOT__` 占位符。

例如：

- `config_file`
- `opts_config_file`
- `conflict_groups_json`

在加载配置时，会自动展开成当前 `scene_graph_service` 目录的真实路径。

所以你拷到远端后，只需要改外部资源路径：

- `img_dir`
- `roidb_file`
- `dict_file`
- `image_file`
- `weight_file`
- `glove_dir`

## 命名空间兼容

当前目录内部新增了：

- [bootstrap_runtime.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/bootstrap_runtime.py)

它负责做两件事：

1. 自动把 `vendor` 加入运行时搜索路径。
2. 自动兼容旧代码里 `services.scene_graph_service` 的导入方式。

因此你把这个目录单独拷走后，直接跑脚本即可，不要求外面一定还套着原仓库的 `services/` 父目录。

## 当前实际验证结果

本地已经用独立化后的路径重新跑过单图入口：

- 输入：`test_000002`
- 输出：成功生成场景图 JSON

这说明：

- 服务目录优先使用 `vendor` 源码是有效的
- 当前单图推理主链路已经不再回头依赖仓库根目录源码

## 当前边界

这里说的“独立”，是指：

- 目录内自带运行源码
- 不依赖根仓库源码
- 不依赖编译 `_C` 扩展

但不代表：

- 所有外部数据文件都被塞进目录里

外部数据和权重仍然是运行时资源，需要你在远端自己挂载或填写路径。

## 远端建议步骤

1. 拷贝整个 `scene_graph_service` 目录到远端。
2. 在远端准备数据、GloVe、权重。
3. 修改 `runtime_config_example.json` 里的外部资源路径。
4. 先跑单图入口。
5. 单图通过后，再跑全量评估入口。

## 当前最重要的一句话

现在这套 `scene_graph_service`，已经可以作为“自带源码的独立服务目录”拿到远端验证。

它当前仍然复用了 `maskrcnn_benchmark` 的源码逻辑，但这些源码已经被内聚到了服务目录内部，而且当前推理链路不再以“必须先编译 `_C` 扩展”为前提。
