# scene_graph_service 关系头迁移进度

## 当前结论

截至现在，`scene_graph_service` 已经完成了关系头内部三块核心模块的服务层迁移，并且都能直接加载现有 HTCL `stage1` 权重：

- `roi_heads.relation.box_feature_extractor`
- `roi_heads.relation.union_feature_extractor`
- `roi_heads.relation.predictor`

目前这三块已经不再依赖 `maskrcnn_benchmark` 运行时。

还没有迁完的是：

- `backbone`
- `rpn`
- 更上游的整图视觉特征提取入口

所以当前状态可以概括为：

- 关系头内部已经基本完成服务层自有化
- 整套 `predcls` 全量评估还差 backbone/FPN 这段接入

## 已落地文件

核心模块：

- [relation_utils_lite.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/modules/relation_utils_lite.py)
- [transformer_lite.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/modules/transformer_lite.py)
- [relation_predictors_lite.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/modules/relation_predictors_lite.py)
- [relation_head_lite.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/modules/relation_head_lite.py)

相关底层模块：

- [box_feature_extractors_lite.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/modules/box_feature_extractors_lite.py)
- [relation_feature_extractors_lite.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/modules/relation_feature_extractors_lite.py)
- [pooler_lite.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/ops/pooler_lite.py)

验证脚本：

- [run_relation_predictor_weight_check.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_relation_predictor_weight_check.py)
- [run_relation_head_weight_check.py](E:/MasterDegree/Experiment/HTCL/services/scene_graph_service/app/scripts/run_relation_head_weight_check.py)

## 已验证结果

### 1. Predictor 权重完全对齐

脚本：

- `run_relation_predictor_weight_check.py`

验证结果：

- `num_predictor_keys_in_checkpoint = 142`
- `missing_keys = []`
- `unexpected_keys = []`

说明：

- 服务层版 `PENET_HTCL` 关系预测器已经可以无漂移地吃进当前 `model_0006000.pth` 里的 predictor 参数

### 2. 整个 relation head 内部三块权重完全对齐

脚本：

- `run_relation_head_weight_check.py`

验证结果：

- `box_feature_extractor.missing_keys = []`
- `box_feature_extractor.unexpected_keys = []`
- `union_feature_extractor.missing_keys = []`
- `union_feature_extractor.unexpected_keys = []`
- `predictor.missing_keys = []`
- `predictor.unexpected_keys = []`

说明：

- 现在服务层已经可以完整承接 `roi_heads.relation.*` 这一层的权重

## 本地验证命令

### 1. 验证 predictor

```powershell
D:\anaconda\envs\cssmx\python.exe services\scene_graph_service\app\scripts\run_relation_predictor_weight_check.py `
  --base-config projects\construction_predcls\configs\construction_relation_base.yaml `
  --opts-config projects\construction_predcls\configs\construction_htcl_predcls_opts.yaml `
  --weight-file results\precls\ours\model_0006000.pth `
  --img-dir datasets\construction_vg_raw_train_test_full_20260703\vg\VG_100K `
  --roidb-file datasets\construction_vg_raw_train_test_full_20260703\vg\VG-SGG-with-attri.h5 `
  --dict-file datasets\construction_vg_raw_train_test_full_20260703\vg\VG-SGG-dicts-with-attri.json `
  --image-file datasets\construction_vg_raw_train_test_full_20260703\vg\image_data.json `
  --glove-dir datasets\construction_vg_raw_train_test_full_20260703\vg
```

### 2. 验证整个 relation head

```powershell
D:\anaconda\envs\cssmx\python.exe services\scene_graph_service\app\scripts\run_relation_head_weight_check.py `
  --weight-file results\precls\ours\model_0006000.pth `
  --img-dir datasets\construction_vg_raw_train_test_full_20260703\vg\VG_100K `
  --roidb-file datasets\construction_vg_raw_train_test_full_20260703\vg\VG-SGG-with-attri.h5 `
  --dict-file datasets\construction_vg_raw_train_test_full_20260703\vg\VG-SGG-dicts-with-attri.json `
  --image-file datasets\construction_vg_raw_train_test_full_20260703\vg\image_data.json `
  --glove-dir datasets\construction_vg_raw_train_test_full_20260703\vg `
  --conflict-groups-json projects\construction_predcls\config\construction_conflict_groups_v2_near_nextto.json
```

## 下一步建议

下一步不要再扩展旧的 `legacy_compat` 脚本，而是继续沿着服务层主线推进：

1. 迁移 `backbone + FPN`，先做到能给 `RelationHeadLite` 提供真实多尺度特征图。
2. 在 `HTCLServiceRunner` 里用服务层版 relation head 替换当前的 `NotImplementedError`。
3. 接上已有 evaluator，跑全量 `test`，对比原主线 `predcls` 指标是否漂移。

## 这一阶段的判断

这一轮最关键的进展不是“先把指标跑出来”，而是已经确认：

- 关系头不是黑盒
- 关系头不是强绑定 `maskrcnn_benchmark`
- 关系头内部核心权重已经可以被服务层自有模块稳定承接

这意味着后面真正难点已经被收敛到一个更明确的范围：

- 主要剩下视觉 backbone/FPN 迁移
- 而不是整个 HTCL 关系头都得重写
