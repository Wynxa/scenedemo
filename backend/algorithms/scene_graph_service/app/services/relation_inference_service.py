from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from services.scene_graph_service.app.adapters.htcl_runtime_adapter import (
    HtclRuntimeAdapter,
    LegacyHtclConfig,
)
from services.scene_graph_service.app.postprocess.scene_graph_builder import build_scene_graph_payload


@dataclass
class RelationInferenceRequest:
    image_path: str
    objects: list[dict[str, Any]]
    image_id: str | None = None


class RelationInferenceService:
    def __init__(self, runtime_config: LegacyHtclConfig) -> None:
        self.adapter = HtclRuntimeAdapter(runtime_config)
        self.runtime_config = runtime_config

    def infer(self, request: RelationInferenceRequest) -> dict[str, Any]:
        image_id = request.image_id or os.path.splitext(os.path.basename(request.image_path))[0]
        result = self.adapter.infer(request.image_path, request.objects)
        return build_scene_graph_payload(
            image_id=image_id,
            image_path=request.image_path,
            objects=result["objects"],
            relationships=result["relationships"],
            meta={
                "task": "scene_graph_service_inference",
                "runtime": "legacy_htcl_adapter",
                "config_file": self.runtime_config.config_file,
                "weight_file": self.runtime_config.weight_file,
                "label_dict": self.runtime_config.label_dict,
                "device": self.runtime_config.device,
            },
        )
