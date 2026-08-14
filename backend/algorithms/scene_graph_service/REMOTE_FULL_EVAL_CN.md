# scene_graph_service 远端全量评估说明

## 目标

这份说明对应当前已经跑通的混合链路：

- 视觉特征：`legacy backbone + FPN`
- 关系头：`scene_graph_service` 自有实现

远端评估的目标是：

- 对 `test` 全量图像生成预测
- 用服务层 evaluator 输出完整关系指标

## 需要准备的文件

远端至少需要：

1. `scene_graph_service` 整个目录
2. 关系权重 `model_0006000.pth`
3. 数据目录
4. `VG_100K`
5. `VG-SGG-with-attri.h5`
6. `VG-SGG-dicts-with-attri.json`
7. `image_data.json`
8. `glove.6B.300d.pt`

## 第一步：修改运行时配置

以 [runtime_config_example.json](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/runtime_config_example.json) 为模板，复制出一个远端版本，例如：

- `runtime_config_gpu.json`

或者直接使用已经准备好的模板：

- [runtime_config_gpu_template.json](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/runtime_config_gpu_template.json)

你只需要改这些外部资源路径：

- `img_dir`
- `roidb_file`
- `dict_file`
- `image_file`
- `weight_file`
- `glove_dir`

内部配置路径：

- `config_file`
- `opts_config_file`
- `conflict_groups_json`

当前已经使用 `__SERVICE_ROOT__` 自动展开，不用你手工改到仓库外部。

## 第二步：直接跑全量评估

新增的全量入口脚本：

- [run_htcl_service_full_eval.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_htcl_service_full_eval.py)

远端命令：

```bash
python app/scripts/run_htcl_service_full_eval.py \
  --runtime-config runtime_config_gpu.json \
  --split test \
  --num-im -1 \
  --report-json /root/scene_graph_service_eval_report.json
```

如果你想把预测结果存到指定目录：

```bash
python app/scripts/run_htcl_service_full_eval.py \
  --runtime-config runtime_config_gpu.json \
  --split test \
  --num-im -1 \
  --prediction-dir /root/scene_graph_service_predictions_test \
  --report-json /root/scene_graph_service_eval_report.json
```

## 输出内容

这个脚本会做两件事：

1. 先跑 `HTCLServiceRunner.predict_dataset(...)`
2. 再调用服务层 evaluator 输出完整报告

最终你会拿到：

- 控制台上的完整评估文本
- 一个预测目录，里面每张图一个 JSON
- 一个 `report_json`

## 重点关注的指标

你最终要比对的是这些：

- `R @ 20/50/100`
- `mR @ 20/50/100`
- `ng-R @ 20/50/100`
- `ng-mR @ 20/50/100`
- `zR @ 20/50/100`
- `ng-zR @ 20/50/100`
- `A @ 20/50/100`

## 如果你只想先抽样验证

可以先只跑前几张：

```bash
python app/scripts/run_htcl_service_full_eval.py \
  --runtime-config runtime_config_gpu.json \
  --split test \
  --num-im 10
```

确认：

- 没有路径报错
- 没有 CUDA / 依赖报错
- 能正常输出预测 JSON

然后再上全量。

## 当前结论

对于你现在的目标，远端最直接的验证命令就是：

```bash
python app/scripts/run_htcl_service_full_eval.py \
  --runtime-config runtime_config_gpu.json \
  --split test \
  --num-im -1
```

这条命令就是当前“全量数据 + 最终评估指标”的正式入口。
