from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SceneGraphPredictionResult:
    image_id: str
    image_path: str
    objects: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class SceneGraphPredictionBundle:
    predictions: list[SceneGraphPredictionResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class SceneGraphRunner(Protocol):
    def predict_dataset(self, *, split: str, num_im: int = -1) -> SceneGraphPredictionBundle:
        ...
