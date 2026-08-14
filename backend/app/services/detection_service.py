"""
Detection service: loads registered YOLO models and runs inference.

Accepts a model_id, loads the corresponding YOLO checkpoint from
ModelRegistry, and returns per-image detection results (bbox, label,
confidence) for downstream use or direct API response.
"""

import json
import os
from typing import Any, Dict, List

from flask import current_app
from ultralytics import YOLO


# In-memory model cache: {model_id: (YOLO instance, config_dict)}
_MODEL_CACHE: Dict[int, tuple] = {}


def _get_model_from_registry(model_id: int):
    """Fetch model record from the database."""
    from app.models.model_registry import ModelRegistry

    model = ModelRegistry.query.get(model_id)
    if model is None:
        raise ValueError(f"模型 #{model_id} 不存在")
    if model.service_type != "yolo_detection":
        raise ValueError(f"模型 #{model_id} 不是 YOLO 检测模型 (当前类型: {model.service_type})")
    return model


def _load_yolo(model_id: int):
    """Load (or retrieve from cache) a YOLO model by registry id."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    record = _get_model_from_registry(model_id)
    ckpt = record.checkpoint_path

    if not os.path.isabs(ckpt):
        ckpt = os.path.join(current_app.config["CHECKPOINT_FOLDER"], ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {ckpt}")

    yolo = YOLO(ckpt)
    config = json.loads(record.config_json) if record.config_json else {}
    _MODEL_CACHE[model_id] = (yolo, config)
    return _MODEL_CACHE[model_id]


def clear_model_cache(model_id: int = None):
    """Clear the in-memory model cache (useful after model update)."""
    if model_id is not None:
        _MODEL_CACHE.pop(model_id, None)
    else:
        _MODEL_CACHE.clear()


def detect_with_model(model_id: int, image_path: str) -> List[Dict[str, Any]]:
    """Run YOLO detection on a single image file.

    Returns a list of detection dicts, each containing:
      - labelName, labelId, bbox [x1,y1,x2,y2], confidence
    """
    yolo, config = _load_yolo(model_id)

    conf = float(config.get("confidenceThreshold", 0.25))
    iou = float(config.get("iouThreshold", 0.45))
    img_sz = int(config.get("imageSize", 640))

    results = yolo(image_path, conf=conf, iou=iou, imgsz=img_sz, verbose=False)

    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            label = results[0].names.get(cls_id, str(cls_id))

            detections.append({
                "labelName": label,
                "labelId": cls_id,
                "bbox": [round(v, 2) for v in xyxy],
                "confidence": round(conf_val, 4),
            })

    return detections


# Keep the old mock interface for backward-compatible label maps
BEHAVIOR_CLASSES = [
    "climbing_scaffold_frame", "leaning_out", "standing_on_guardrail",
    "throwing_material", "leaning_outside_platform", "crossing_guardrail",
    "working_outside_guardrail", "climbing_cross_brace",
    "unsafe_step_off", "missing_step", "unstable_posture", "throwing_objects",
]

BEHAVIOR_LABELS = {
    "climbing_scaffold_frame": "攀爬脚手架框架",
    "leaning_out": "身体探出",
    "standing_on_guardrail": "站立在护栏上",
    "throwing_material": "抛掷物料",
    "leaning_outside_platform": "平台外探身",
    "crossing_guardrail": "翻越护栏",
    "working_outside_guardrail": "护栏外作业",
    "climbing_cross_brace": "攀爬交叉支撑",
    "unsafe_step_off": "不安全下步",
    "missing_step": "踏空",
    "unstable_posture": "不稳定姿势",
    "throwing_objects": "抛物行为",
}


def detect_single_image(file):
    """Legacy single-image detection (kept for /detection/single compatibility).
    Uses the active model from registry if available, otherwise falls back to mock.
    """
    from app.models.model_registry import ModelRegistry

    active = ModelRegistry.query.filter_by(status="active").first()
    if active and active.service_type == "yolo_detection":
        from app.utils.file_utils import save_uploaded_file
        import os as _os

        rel = save_uploaded_file(file)
        full = _os.path.join(current_app.config["UPLOAD_FOLDER"], rel)
        dets = detect_with_model(active.id, full)
        return {
            "image_name": file.filename,
            "modelName": active.name,
            "modelId": active.id,
            "totalWorkers": len(dets),
            "unsafeCount": sum(1 for d in dets if d["labelName"] != "helmet"),
            "detections": dets,
        }

    # Fallback to mock for backward compatibility
    return _mock_detection(file.filename)


def _mock_detection(image_name):
    import random
    seed = hash(image_name) % 10000
    rng = random.Random(seed)
    num = rng.randint(0, 5)
    dets = []
    for _ in range(num):
        dets.append({
            "workerBbox": f"{rng.randint(10,800)},{rng.randint(10,600)},{rng.randint(60,200)},{rng.randint(120,400)}",
            "behaviorProbs": {},
            "riskProbs": {},
            "riskSeverity": round(rng.random(), 4),
            "primaryBehavior": "safe",
            "primaryBehaviorLabel": "安全",
            "confidence": round(rng.random(), 4),
        })
    return {
        "image_name": image_name,
        "totalWorkers": len(dets),
        "unsafeCount": 0,
        "detections": dets,
    }
