from __future__ import annotations

from pathlib import Path

from services.hazard_reasoning_service.app.config import (
    HazardModelConfig,
    HazardReasoningRuntimeConfig,
    HazardRuntimePathsConfig,
)
from services.hazard_reasoning_service.app.io.json_io import load_json


def load_runtime_config(path: str | Path) -> HazardReasoningRuntimeConfig:
    payload = load_json(Path(path))
    return HazardReasoningRuntimeConfig(
        paths=HazardRuntimePathsConfig(**payload.get("paths", {})),
        model=HazardModelConfig(**payload.get("model", {})),
        device=payload.get("device", "cpu"),
    )
