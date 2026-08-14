from __future__ import annotations

import torch
from torchvision.ops import roi_align


class ROIAlignLite(torch.nn.Module):
    def __init__(
        self,
        output_size: tuple[int, int] | int,
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input_tensor: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return roi_align(
            input=input_tensor,
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )
