from __future__ import annotations

import torch
from torchvision.ops import nms


def nms_lite(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    return nms(boxes, scores, iou_threshold)
