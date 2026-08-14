from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from services.scene_graph_service.app.runner.checkpoint_mapping import (
    map_checkpoint_keys_to_service_modules,
)


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise TypeError("Unable to locate a state_dict-like mapping in checkpoint payload.")


def inspect_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    checkpoint_path = str(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _unwrap_state_dict(payload)
    key_list = list(state_dict.keys())

    prefix_counter: Counter[str] = Counter()
    for key in key_list:
        prefix = key.split(".", 1)[0]
        prefix_counter[prefix] += 1

    tensor_items = []
    total_parameters = 0
    for key, value in state_dict.items():
        if hasattr(value, "numel"):
            numel = int(value.numel())
            total_parameters += numel
            tensor_items.append(
                {
                    "name": key,
                    "shape": list(value.shape),
                    "numel": numel,
                    "dtype": str(value.dtype),
                }
            )

    return {
        "checkpoint_path": checkpoint_path,
        "num_keys": len(key_list),
        "total_parameters": total_parameters,
        "top_level_prefixes": dict(prefix_counter.most_common()),
        "sample_tensors": tensor_items[:20],
        "service_module_mapping": map_checkpoint_keys_to_service_modules(key_list),
    }
