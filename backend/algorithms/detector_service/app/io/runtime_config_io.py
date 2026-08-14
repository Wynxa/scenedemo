from __future__ import annotations

from pathlib import Path

from services.detector_service.app.config import (
    DetectorModelRuntimeConfig,
    DetectorServiceRuntimeConfig,
)
from services.detector_service.app.io.json_io import load_json


def load_runtime_config(path: str | Path) -> DetectorServiceRuntimeConfig:
    payload = load_json(path)
    models = {
        key: DetectorModelRuntimeConfig(**value)
        for key, value in payload.get("models", {}).items()
    }
    return DetectorServiceRuntimeConfig(
        device=payload.get("device", "cpu"),
        iou_nms=float(payload.get("iou_nms", 0.5)),
        models=models,
    )
