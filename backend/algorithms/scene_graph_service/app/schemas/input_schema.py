from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DetectionObjectInput:
    object_id: int
    label: Optional[str] = None
    label_id: Optional[int] = None
    bbox: List[float] = field(default_factory=list)
    score: float = 1.0


@dataclass
class SceneGraphRequest:
    image_id: Optional[str]
    image_path: str
    objects: List[DetectionObjectInput] = field(default_factory=list)
