from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from app.services.algorithm_runtime import load_algorithm_runtime_config


def _ensure_algorithm_sys_path() -> Path:
    backend_root = Path(__file__).resolve().parents[2]
    algorithms_root = backend_root / "algorithms"
    path_str = str(algorithms_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return algorithms_root


class AlgorithmServiceAdapters:
    def __init__(self) -> None:
        self.runtime = load_algorithm_runtime_config()
        self.algorithms_root = _ensure_algorithm_sys_path()
        self._detector_service = None
        self._scene_graph_runner = None
        self._hazard_service = None

    def run_detector(self, image_path: str, image_id: str) -> dict[str, Any]:
        from detector_service.app.config import DetectorModelRuntimeConfig, DetectorServiceRuntimeConfig
        from detector_service.app.services.detector_service import LayeredDetectorService

        if self._detector_service is None:
            model_configs = {
                key: DetectorModelRuntimeConfig(**value)
                for key, value in self.runtime.detector.get("runtime_config", {}).get("models", {}).items()
            }
            runtime_config = DetectorServiceRuntimeConfig(
                device=self.runtime.detector.get("runtime_config", {}).get("device", "cpu"),
                iou_nms=float(self.runtime.detector.get("runtime_config", {}).get("iou_nms", 0.5)),
                models=model_configs,
            )
            self._detector_service = LayeredDetectorService(runtime_config)
        return self._detector_service.infer_image(image_path=image_path, image_id=image_id)

    def run_scene_graph(self, image_path: str, image_id: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
        from scene_graph_service.app.io.runtime_config_io import runtime_config_from_dict
        from scene_graph_service.app.runner.htcl_service_runner import HTCLServiceRunner

        if self._scene_graph_runner is None:
            runtime_config = runtime_config_from_dict(self.runtime.scene_graph.get("runtime_config", {}))
            self._scene_graph_runner = HTCLServiceRunner(runtime_config)
        result = self._scene_graph_runner.predict_request(image_id=image_id, image_path=image_path, objects=objects)
        return {
            "image_id": result.image_id,
            "image_path": result.image_path,
            "objects": result.objects,
            "relationships": result.relationships,
            "meta": result.meta,
        }

    def run_hazard_reasoning(self, scene_graph_payload: dict[str, Any]) -> dict[str, Any]:
        from hazard_reasoning_service.app.io.runtime_config_io import load_runtime_config as load_hazard_runtime_config
        from hazard_reasoning_service.app.services.hazard_inference_service import HazardReasoningService

        if self._hazard_service is None:
            backend_root = Path(__file__).resolve().parents[2]
            temp_runtime = backend_root / "instance" / "algorithm_runtime.reasoning.tmp.json"
            temp_runtime.write_text(
                json.dumps(self.runtime.hazard_reasoning.get("runtime_config", {}), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_config = load_hazard_runtime_config(temp_runtime)
            self._hazard_service = HazardReasoningService(runtime_config)
        return self._hazard_service.infer_from_scene_graph(scene_graph_payload)

    def run_full_pipeline(self, image_path: str, image_id: str) -> dict[str, Any]:
        detector_result = self.run_detector(image_path=image_path, image_id=image_id)
        scene_graph_result = self.run_scene_graph(
            image_path=detector_result["image_path"],
            image_id=detector_result["image_id"],
            objects=detector_result["objects"],
        )
        hazard_result = self.run_hazard_reasoning(scene_graph_result)
        return {
            "detector_result": detector_result,
            "scene_graph_result": scene_graph_result,
            "hazard_result": hazard_result,
        }


_adapter_instance: AlgorithmServiceAdapters | None = None
_adapter_lock = threading.Lock()


def get_algorithm_adapters() -> AlgorithmServiceAdapters:
    """Return one warm adapter per backend process.

    Its child services cache their model weights, avoiding a full reload for
    every image submitted to the CPU-only runtime.
    """
    global _adapter_instance
    if _adapter_instance is None:
        with _adapter_lock:
            if _adapter_instance is None:
                _adapter_instance = AlgorithmServiceAdapters()
    return _adapter_instance
