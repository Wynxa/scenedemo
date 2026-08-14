from dataclasses import dataclass, field


@dataclass
class DetectorModelRuntimeConfig:
    name: str = ""
    repo_root: str = ""
    weight_path: str = ""
    conf: float = 0.25
    imgsz: int = 1280
    target_labels: list[str] = field(default_factory=list)


@dataclass
class DetectorServiceRuntimeConfig:
    device: str = "cpu"
    iou_nms: float = 0.5
    models: dict[str, DetectorModelRuntimeConfig] = field(default_factory=dict)
