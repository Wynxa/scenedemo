from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SceneGraphServiceConfig:
    service_name: str = "scene_graph_service"
    debug: bool = False
    project_root: Path | None = None


@dataclass
class InferencePreprocessConfig:
    min_size: int = 600
    max_size: int = 1000
    pixel_mean: list[float] = field(default_factory=lambda: [102.9801, 115.9465, 122.7717])
    pixel_std: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    to_bgr255: bool = True


@dataclass
class RuntimePathsConfig:
    img_dir: str = ""
    roidb_file: str = ""
    dict_file: str = ""
    image_file: str = ""
    weight_file: str = ""
    config_file: str = ""
    opts_config_file: str = ""
    glove_dir: str = ""
    conflict_groups_json: str = ""


@dataclass
class ServiceOwnedRuntimeConfig:
    paths: RuntimePathsConfig = field(default_factory=RuntimePathsConfig)
    preprocess: InferencePreprocessConfig = field(default_factory=InferencePreprocessConfig)
    device: str = "cpu"
    mode: str = "predcls"
    min_rel_score: float = 0.0
    topk_relations: int = 0
