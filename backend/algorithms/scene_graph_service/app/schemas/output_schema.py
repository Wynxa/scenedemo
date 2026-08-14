from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SceneObjectOutput:
    object_index: int
    object_id: int
    name: str
    label_id: int
    x: float
    y: float
    w: float
    h: float
    bbox_xyxy: List[float] = field(default_factory=list)
    score: float = 0.0
    det_score: Optional[float] = None


@dataclass
class SceneRelationshipOutput:
    relationship_id: int
    subject_index: int
    object_index: int
    subject_id: int
    object_id: int
    subject_name: str
    object_name: str
    predicate_id: int
    predicate: str
    score: float


@dataclass
class SceneGraphResponse:
    image_id: str
    file_name: str
    image_path: str
    objects: List[SceneObjectOutput] = field(default_factory=list)
    relationships: List[SceneRelationshipOutput] = field(default_factory=list)
