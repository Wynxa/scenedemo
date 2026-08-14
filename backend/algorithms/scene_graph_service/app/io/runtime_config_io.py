from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scene_graph_service.bootstrap_runtime import get_service_root
from services.scene_graph_service.app.config import (
    InferencePreprocessConfig,
    RuntimePathsConfig,
    ServiceOwnedRuntimeConfig,
)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping in runtime config payload.")
    return value


def _expand_path_token(value: str) -> str:
    return value.replace("__SERVICE_ROOT__", str(get_service_root()))


def load_runtime_config(config_path: str | Path) -> ServiceOwnedRuntimeConfig:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return runtime_config_from_dict(payload)


def runtime_config_from_dict(payload: dict[str, Any]) -> ServiceOwnedRuntimeConfig:
    paths_payload = _as_dict(payload.get("paths"))
    preprocess_payload = _as_dict(payload.get("preprocess"))
    return ServiceOwnedRuntimeConfig(
        paths=RuntimePathsConfig(
            img_dir=_expand_path_token(str(paths_payload.get("img_dir", ""))),
            roidb_file=_expand_path_token(str(paths_payload.get("roidb_file", ""))),
            dict_file=_expand_path_token(str(paths_payload.get("dict_file", ""))),
            image_file=_expand_path_token(str(paths_payload.get("image_file", ""))),
            weight_file=_expand_path_token(str(paths_payload.get("weight_file", ""))),
            config_file=_expand_path_token(str(paths_payload.get("config_file", ""))),
            opts_config_file=_expand_path_token(str(paths_payload.get("opts_config_file", ""))),
            glove_dir=_expand_path_token(str(paths_payload.get("glove_dir", ""))),
            conflict_groups_json=_expand_path_token(str(paths_payload.get("conflict_groups_json", ""))),
        ),
        preprocess=InferencePreprocessConfig(
            min_size=int(preprocess_payload.get("min_size", 600)),
            max_size=int(preprocess_payload.get("max_size", 1000)),
            pixel_mean=[float(v) for v in preprocess_payload.get("pixel_mean", [102.9801, 115.9465, 122.7717])],
            pixel_std=[float(v) for v in preprocess_payload.get("pixel_std", [1.0, 1.0, 1.0])],
            to_bgr255=bool(preprocess_payload.get("to_bgr255", True)),
        ),
        device=str(payload.get("device", "cpu")),
        mode=str(payload.get("mode", "predcls")),
        min_rel_score=float(payload.get("min_rel_score", 0.0)),
        topk_relations=int(payload.get("topk_relations", 0)),
    )


def dump_runtime_config(runtime_config: ServiceOwnedRuntimeConfig) -> dict[str, Any]:
    return {
        "paths": {
            "img_dir": runtime_config.paths.img_dir,
            "roidb_file": runtime_config.paths.roidb_file,
            "dict_file": runtime_config.paths.dict_file,
            "image_file": runtime_config.paths.image_file,
            "weight_file": runtime_config.paths.weight_file,
            "config_file": runtime_config.paths.config_file,
            "opts_config_file": runtime_config.paths.opts_config_file,
            "glove_dir": runtime_config.paths.glove_dir,
            "conflict_groups_json": runtime_config.paths.conflict_groups_json,
        },
        "preprocess": {
            "min_size": runtime_config.preprocess.min_size,
            "max_size": runtime_config.preprocess.max_size,
            "pixel_mean": runtime_config.preprocess.pixel_mean,
            "pixel_std": runtime_config.preprocess.pixel_std,
            "to_bgr255": runtime_config.preprocess.to_bgr255,
        },
        "device": runtime_config.device,
        "mode": runtime_config.mode,
        "min_rel_score": runtime_config.min_rel_score,
        "topk_relations": runtime_config.topk_relations,
    }
