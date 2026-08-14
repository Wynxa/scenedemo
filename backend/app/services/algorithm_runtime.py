from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AlgorithmRuntimeConfig:
    backend_root: Path
    raw: dict[str, Any]

    @property
    def detector(self) -> dict[str, Any]:
        return self.raw.get("detector", {})

    @property
    def scene_graph(self) -> dict[str, Any]:
        return self.raw.get("scene_graph", {})

    @property
    def hazard_reasoning(self) -> dict[str, Any]:
        return self.raw.get("hazard_reasoning", {})


def _expand_tokens(value: Any, backend_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace("__BACKEND_ROOT__", str(backend_root)).replace("__PROJECT_ROOT__", str(backend_root.parent))
    if isinstance(value, list):
        return [_expand_tokens(v, backend_root) for v in value]
    if isinstance(value, dict):
        return {k: _expand_tokens(v, backend_root) for k, v in value.items()}
    return value


def load_algorithm_runtime_config(config_path: str | Path | None = None) -> AlgorithmRuntimeConfig:
    backend_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = backend_root / "instance" / "algorithm_runtime.json"
    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    payload = _expand_tokens(payload, backend_root)
    return AlgorithmRuntimeConfig(backend_root=backend_root, raw=payload)
