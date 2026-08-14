# detector_service

这个目录是第一个服务，也就是目标检测服务。

## 当前定位

这个服务专门负责一张上传图像的目标检测，不做关系预测，也不做危险行为推理。

当前采用分层策略：

- `stw_ppe`
  - 权重：`E:/MasterDegree/scenedemo/weight/stw/best.pt`
  - 负责类：`helmet`、`gloves`、`vest`
- `yolo_general`
  - 权重：`E:/MasterDegree/scenedemo/weight/yolo/best.pt`
  - 负责类：`worker`、`excavator`、`scaffold`、`rebar_zone`

也就是说，这个服务不是两个独立接口，而是一个统一检测入口，内部根据“模型负责的类”做路由，再合并结果。

## 为什么这样注册最合适

因为你的业务语义已经很清楚：

- PPE 小目标用 `STW-YOLO`
- 场景主体目标用普通 `YOLO`

所以最稳妥的注册方式不是“前端选模型”，而是：

- 注册成一个服务实例：`layered_detector_service`
- 服务内部挂两个模型实例：
  - `stw_ppe`
  - `yolo_general`
- 对外只暴露一个推理入口

这样后面前端上传一张图，后端只调一次检测服务即可。

## 推荐注册方式

建议把第一个服务注册成一个“服务”，不是两个分开的正式业务接口。

服务名建议：

- `detector_service`

内部实例：

- `stw_ppe`
- `yolo_general`

对外统一入口：

- `POST /api/infer/detector`

这样返回值稳定，也最方便接第二个服务。

## 统一输出

这个服务输出统一 JSON，直接兼容第二个服务：

```json
{
  "image_id": "004910",
  "image_path": "E:/path/to/004910.jpg",
  "width": 4000,
  "height": 3000,
  "objects": [
    {
      "object_id": 0,
      "label_id": 11,
      "label": "worker",
      "bbox": [100.0, 120.0, 300.0, 800.0],
      "score": 0.97,
      "source": "yolo_general",
      "model_name": "best"
    }
  ],
  "meta": {
    "detector_service": "layered_detector_service",
    "detector_strategy": "stw_ppe + yolo_general",
    "device": "cpu",
    "num_objects": 1
  }
}
```

`scene_graph_service` 已经能吃这种结构，不需要你再改第二个服务协议。

## 本地环境

你已经说明 `cssmx` 环境里有 `torch` 和 `yolo`，所以当前建议就是：

- 第一个服务先直接跑在 `cssmx`
- 暂时不拆独立容器
- 先把推理脚本验证通

## 运行配置

配置文件参考 [runtime_config_example.json](/E:/MasterDegree/Experiment/HTCL/services/detector_service/runtime_config_example.json)。

你主要改这几项：

- `models.stw_ppe.repo_root`
- `models.stw_ppe.weight_path`
- `models.yolo_general.weight_path`
- `device`

如果后面 GPU 可用，把 `device` 改成 `cuda:0` 即可。

## 单图测试命令

```powershell
D:/anaconda/envs/cssmx/python.exe -m services.detector_service.app.scripts.run_detector_infer `
  --runtime-config E:/MasterDegree/Experiment/HTCL/services/detector_service/runtime_config_example.json `
  --image-path E:/MasterDegree/Experiment/HTCL/004910.jpg `
  --image-id 004910 `
  --output-json E:/MasterDegree/Experiment/HTCL/tmp/detector_004910.json
```

## 后续在总流程里的位置

完整链路保持不变：

1. `detector_service`
2. `scene_graph_service`
3. `hazard_reasoning_service`

正式业务页只调：

- `POST /api/infer/pipeline`

但后端内部就是三段串行。
