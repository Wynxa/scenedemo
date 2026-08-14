# 场景图关系推理服务目录说明

## 1. 当前阶段目标

这一阶段的目标很明确：

- 只服务于推理
- 不覆盖训练流程
- 新开一个独立文件夹
- 作为整套关系推理服务的后端代码根目录
- 后续这个文件夹可以单独迁移、单独封装、单独打包

也就是说，这里不是对整个 HTCL 仓库做大重构，而是先落一个面向 demo 的 `scene_graph_service`：

- 输入面向 `image + bbox + label`
- 输出面向 `relationships + scene_graph`

## 2. 当前目录职责

这个目录现在承担三类职责：

- 服务侧输入输出协议
- 推理链路编排
- 逐步替换 `maskrcnn_benchmark` 依赖的重构落点

## 3. 目录结构

```text
scene_graph_service/
  README_CN.md
  requirements.in
  app/
    __init__.py
    config.py
    schemas/
      __init__.py
      input_schema.py
      output_schema.py
    io/
      __init__.py
      json_io.py
    structures/
      __init__.py
      boxlist_lite.py
    transforms/
      __init__.py
      image_transforms.py
    ops/
      __init__.py
      roi_align_ops.py
      nms_ops.py
      pooler_lite.py
    postprocess/
      __init__.py
      relation_postprocess.py
      scene_graph_builder.py
    adapters/
      __init__.py
      legacy_boxlist_bridge.py
      htcl_runtime_adapter.py
    services/
      __init__.py
      request_preprocessor.py
      relation_inference_service.py
    api/
      __init__.py
      scene_graph_api.py
    scripts/
      __init__.py
      run_scene_graph_infer.py
```

## 4. 分层说明

### 4.1 `schemas`

定义服务输入输出字段约定：

- 输入：图像路径、对象框、类别
- 输出：对象、关系、场景图

### 4.2 `io`

负责 JSON 读写和协议组织。

### 4.3 `structures`

负责轻量数据结构实现，目前重点是：

- `BoxListLite`

### 4.4 `transforms`

负责：

- 图像 resize
- normalize
- bbox 同步变换

### 4.5 `ops`

负责替换旧框架的经典算子：

- `ROIAlign`
- `NMS`
- `Pooler`

### 4.6 `postprocess`

负责：

- relation 输出整理
- triple 排序
- 场景图构建

### 4.7 `adapters`

负责过渡期桥接已有 HTCL 推理能力。

当前阶段，`htcl_runtime_adapter.py` 仍然依赖旧的 HTCL 模型构建和权重加载逻辑，但它的输入组织已经开始切到新服务目录内部实现。

### 4.8 `services`

负责整条推理链编排：

- 输入预处理
- 调模型
- 后处理
- 返回标准场景图

### 4.9 `api`

负责后续 Flask 接口封装。

### 4.10 `scripts`

负责命令行调试入口。

## 5. 当前阶段边界

这一阶段先不做：

- 训练逻辑迁移
- stage1 / stage2 训练入口改造
- 整套模型组件完全重写

这一阶段只做：

- 推理服务目录独立
- 服务链路独立
- 输入输出协议独立
- 为后续逐步去掉 `maskrcnn_benchmark` 做工程落点

## 6. 当前已完成的依赖收缩

当前目录已经完成两层拆分：

1. 服务侧自己的输入预处理

   - `request_preprocessor.py`
   - `boxlist_lite.py`
   - `image_transforms.py`
2. 底层经典算子替换壳

   - `roi_align_ops.py`
   - `nms_ops.py`
   - `pooler_lite.py`

当前仍保留的旧依赖，主要集中在：

- `maskrcnn_benchmark.config`
- `build_detection_model`
- `DetectronCheckpointer`
- 旧关系模型前向主干

也就是说，现在已经把旧依赖压缩到 adapter 这一层，后续继续替换时，改动范围会更可控。

## 7. 推荐推进顺序

建议继续按下面顺序推进：

1. 保持 `scene_graph_service` 作为唯一新增目录
2. 继续把 adapter 之外的逻辑迁到本目录
3. 保证 CLI 能稳定跑通
4. 再逐步替换旧模型入口和关系头依赖
5. 最后再接 Flask API

## 8. 当前 CLI 入口

当前命令行入口：

- [run_scene_graph_infer.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_scene_graph_infer.py)
- [run_dataset_eval_legacy_compat.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_dataset_eval_legacy_compat.py)

示例：

```powershell
python E:\MasterDegree\Experiment\HTCL\services\scene_graph_service\app\scripts\run_scene_graph_infer.py `
  --config-file E:\MasterDegree\Experiment\HTCL\configs\e2e_relation_X_101_32_8_FPN_1x.yaml `
  --weight-file E:\MasterDegree\Experiment\HTCL\results\precls\vctree_htcl_stage1\model_final.pth `
  --input-json E:\MasterDegree\Experiment\HTCL\tmp\htcl_input.json `
  --output-json E:\MasterDegree\Experiment\HTCL\tmp\scene_graph_service_output.json `
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True `
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```

说明：

- 这条命令现在仍然通过 `htcl_runtime_adapter.py` 调底层 HTCL
- 但服务侧输入预处理和场景图输出，已经切到 `scene_graph_service` 自己的代码里
- 这就是后续接 Flask `pipeline / scene-graph / reasoning` 接口的基础目录

`h5` 整集评估入口示例：

```powershell
python E:\MasterDegree\Experiment\HTCL\services\scene_graph_service\app\scripts\run_dataset_eval_legacy_compat.py `
  --config-file E:\MasterDegree\Experiment\HTCL\projects\construction_predcls\configs\construction_relation_base.yaml `
  --my-opts E:\MasterDegree\Experiment\HTCL\projects\construction_predcls\configs\construction_htcl_predcls_opts.yaml `
  --output-dir E:\MasterDegree\Experiment\HTCL\tmp\scene_graph_service_eval `
  MODEL.WEIGHT E:\MasterDegree\Experiment\HTCL\results\precls\ours\model_0006000.pth `
  MODEL.PRETRAINED_DETECTOR_CKPT E:\path\to\pretrained_faster_rcnn\model_final.pth `
  GLOVE_DIR E:\path\to\vg_glove `
  MODEL.DEVICE cuda `
  DATASETS.TO_TEST test `
  TEST.IMS_PER_BATCH 1
```

这个脚本的用途不是替代主线评估，而是做“新服务预处理是否漂移”的对照验证：

- 数据输入仍然是主线同一份 `h5`
- 评估指标仍然走主线 `vg_eval`
- 唯一替换的是前向前的输入构造与图像预处理，改用 `scene_graph_service` 里的实现

## 9. 结论

从这一阶段开始，后续“关系推理服务”的新增代码，优先都落在 `scene_graph_service` 目录里，而不是继续散落在 `tools` 或 `maskrcnn_benchmark` 目录中。

当前这套目录已经能作为关系推理服务后端的骨架继续扩展，后面我们只需要沿着 adapter 边界继续收缩旧依赖即可。

```Python
python services/scene_graph_service/app/scripts/run_dataset_eval_legacy_compat.py `
  --config-file projects/construction_predcls/configs/construction_relation_base.yaml `
  --my-opts projects/construction_predcls/configs/construction_htcl_predcls_opts.yaml `
  --output-dir "E:\MasterDegree\Experiment\HTCL\tmp\scene_graph_service_eval_cpu" `
  MODEL.WEIGHT "E:\MasterDegree\Experiment\HTCL\results\precls\ours\model_0006000.pth" `
  MODEL.PRETRAINED_DETECTOR_CKPT "E:\MasterDegree\scenedemo\weight\model_dector.pth" `
  GLOVE_DIR "E:\MasterDegree\Experiment\HTCL\datasets\construction_vg_raw_train_test_full_20260703\vg" `
  MODEL.DEVICE "cpu" `
  DATASETS.TO_TEST test `
  TEST.IMS_PER_BATCH 1
```
