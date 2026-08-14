from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _pick(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _normalize_bbox(item: dict[str, Any]) -> list[float]:
    bbox = _pick(item, "bbox", "bbox_xyxy", "box")
    if bbox is None:
        x = _pick(item, "x")
        y = _pick(item, "y")
        w = _pick(item, "w", "width")
        h = _pick(item, "h", "height")
        if None not in (x, y, w, h):
            return [float(x), float(y), float(x) + float(w), float(y) + float(h)]
        raise ValueError("Detector object must contain bbox/bbox_xyxy/box or x,y,w,h.")
    if len(bbox) != 4:
        raise ValueError("Detector bbox must contain exactly 4 values.")
    return [float(v) for v in bbox]


def _normalize_object(item: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = {
        "object_id": int(_pick(item, "object_id", "id", default=index)),
        "bbox": _normalize_bbox(item),
        "score": float(_pick(item, "score", "conf", "confidence", default=1.0)),
        "source": str(_pick(item, "source", default="detector_service")),
    }
    label_id = _pick(item, "label_id", "class_id", "category_id")
    label_name = _pick(item, "label", "name", "class_name", "category_name")
    if label_id is not None:
        normalized["label_id"] = int(label_id)
    if label_name is not None:
        normalized["label"] = str(label_name)
    if "track_id" in item and item["track_id"] is not None:
        normalized["track_id"] = int(item["track_id"])
    return normalized


@dataclass
class DetectorPayloadAdapter:
    def build_relation_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_path = _pick(payload, "image_path", "img_path", "file_path")
        if not image_path:
            raise ValueError("Detector payload must contain image_path.")
        raw_objects = _pick(payload, "objects", "detections", "boxes", default=[])
        if not isinstance(raw_objects, list):
            raise TypeError("Detector payload objects/detections must be a list.")

        normalized_objects = [_normalize_object(item, idx) for idx, item in enumerate(raw_objects)]
        return {
            "image_id": _pick(payload, "image_id", "id"),
            "image_path": str(image_path),
            "objects": normalized_objects,
            "meta": {
                "detector_service": _pick(payload, "detector_service", "detector_name"),
                "detector_model": _pick(payload, "detector_model", "model_name"),
                "num_objects": len(normalized_objects),
            },
        }
