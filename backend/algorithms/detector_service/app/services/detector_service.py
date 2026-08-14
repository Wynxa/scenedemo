from __future__ import annotations

import os
import sys
import importlib
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
from torch import nn

from services.detector_service.app.config import (
    DetectorModelRuntimeConfig,
    DetectorServiceRuntimeConfig,
)
from services.detector_service.app.schemas.output_schema import CANONICAL_LABEL_TO_ID
from services.detector_service.app.utils.nms import classwise_nms

SERVICE_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_CONFIG_DIR = SERVICE_ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR = SERVICE_ROOT / ".matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))


@dataclass
class LoadedModelBundle:
    config: DetectorModelRuntimeConfig
    model: object


class LayeredDetectorService:
    def __init__(self, runtime_config: DetectorServiceRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self._bundles: dict[str, LoadedModelBundle] = {}
        self._yolo_class_cache: dict[str, object] = {}

    @staticmethod
    def _normalized_repo_root(repo_root: str) -> str:
        if not repo_root:
            return "__installed_ultralytics__"
        return str(Path(repo_root).resolve())

    def _resolve_yolo_class(self, repo_root: str):
        cache_key = self._normalized_repo_root(repo_root)
        cached = self._yolo_class_cache.get(cache_key)
        if cached is not None:
            return cached

        resolved_root = None if cache_key == "__installed_ultralytics__" else cache_key
        if resolved_root is not None and resolved_root not in sys.path:
            sys.path.insert(0, resolved_root)

        # If a custom STW-YOLO ultralytics fork is needed, flush previously imported
        # official ultralytics modules so Python resolves imports from repo_root first.
        if resolved_root is not None:
            for module_name in list(sys.modules.keys()):
                if module_name == "ultralytics" or module_name.startswith("ultralytics."):
                    sys.modules.pop(module_name, None)

        self._ensure_third_party_shims()
        ultralytics_module = importlib.import_module("ultralytics")
        yolo_class = ultralytics_module.YOLO
        self._yolo_class_cache[cache_key] = yolo_class
        return yolo_class

    @staticmethod
    def _ensure_third_party_shims() -> None:
        if "efficientnet_pytorch.model" not in sys.modules:
            model_module = types.ModuleType("efficientnet_pytorch.model")

            class MemoryEfficientSwish(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x * torch.sigmoid(x)

            model_module.MemoryEfficientSwish = MemoryEfficientSwish
            package_module = types.ModuleType("efficientnet_pytorch")
            package_module.model = model_module
            sys.modules["efficientnet_pytorch"] = package_module
            sys.modules["efficientnet_pytorch.model"] = model_module

    def _load_bundle(self, key: str) -> LoadedModelBundle:
        bundle = self._bundles.get(key)
        if bundle is not None:
            return bundle

        config = self.runtime_config.models[key]
        yolo_class = self._resolve_yolo_class(config.repo_root)
        model = yolo_class(config.weight_path)
        bundle = LoadedModelBundle(config=config, model=model)
        self._bundles[key] = bundle
        return bundle

    @staticmethod
    def _image_size(image_path: str | Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return image.width, image.height

    def _predict_with_bundle(self, bundle: LoadedModelBundle, image_path: str | Path) -> List[dict]:
        results = bundle.model.predict(
            source=str(image_path),
            conf=bundle.config.conf,
            imgsz=bundle.config.imgsz,
            device=self.runtime_config.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        objects: List[dict] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(names[class_id])
            if bundle.config.target_labels and label not in bundle.config.target_labels:
                continue
            bbox_xyxy = [float(v) for v in box.xyxy[0].tolist()]
            objects.append(
                {
                    "label": label,
                    "label_id": CANONICAL_LABEL_TO_ID.get(label),
                    "bbox": bbox_xyxy,
                    "score": float(box.conf.item()),
                    "source": bundle.config.name,
                    "model_name": Path(bundle.config.weight_path).stem,
                }
            )
        return objects

    def infer_image(self, image_path: str | Path, image_id: str | None = None) -> Dict:
        image_path = str(Path(image_path))
        image_width, image_height = self._image_size(image_path)
        all_objects: List[dict] = []

        for key in self.runtime_config.models:
            bundle = self._load_bundle(key)
            all_objects.extend(self._predict_with_bundle(bundle, image_path))

        merged_objects = classwise_nms(all_objects, iou_threshold=self.runtime_config.iou_nms)
        normalized_objects = []
        for index, item in enumerate(merged_objects):
            normalized_objects.append(
                {
                    "object_id": index,
                    "label_id": item["label_id"],
                    "label": item["label"],
                    "bbox": item["bbox"],
                    "score": item["score"],
                    "source": item["source"],
                    "model_name": item["model_name"],
                }
            )

        return {
            "image_id": image_id or Path(image_path).stem,
            "image_path": image_path,
            "width": image_width,
            "height": image_height,
            "objects": normalized_objects,
            "meta": {
                "detector_service": "layered_detector_service",
                "detector_strategy": "stw_ppe + yolo_general",
                "device": self.runtime_config.device,
                "num_objects": len(normalized_objects),
            },
        }
