from dataclasses import dataclass, field


@dataclass
class HazardRuntimePathsConfig:
    checkpoint: str = ""
    prototype_inventory: str = ""
    context_policy: str = ""
    model_cache_dir: str = ""


@dataclass
class HazardModelConfig:
    pretrained_model: str = "bert-base-uncased"
    local_files_only: bool = True
    max_length: int = 96
    projection_hidden_dim: int = 512
    projection_dim: int = 256
    dropout: float = 0.1
    share_backbone: bool = True
    scene_text_field: str = "scene_text_structured"
    topk: int = 3


@dataclass
class HazardReasoningRuntimeConfig:
    paths: HazardRuntimePathsConfig = field(default_factory=HazardRuntimePathsConfig)
    model: HazardModelConfig = field(default_factory=HazardModelConfig)
    device: str = "cpu"
