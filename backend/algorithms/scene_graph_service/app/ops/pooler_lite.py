from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from services.scene_graph_service.app.ops.roi_align_ops import ROIAlignLite
from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite


def _cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


class LevelMapper:
    def __init__(self, k_min: int, k_max: int, canonical_scale: int = 224, canonical_level: int = 4, eps: float = 1e-6):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: list[BoxListLite]) -> torch.Tensor:
        scales = torch.sqrt(_cat([boxlist.area() for boxlist in boxlists]))
        target_levels = torch.floor(self.lvl0 + torch.log2(scales / self.s0 + self.eps))
        target_levels = torch.clamp(target_levels, min=self.k_min, max=self.k_max)
        return target_levels.to(torch.int64) - self.k_min


class PoolerLite(torch.nn.Module):
    def __init__(
        self,
        output_size: tuple[int, int],
        scales: tuple[float, ...] | list[float],
        sampling_ratio: int,
        in_channels: int = 512,
        cat_all_levels: bool = False,
        aligned: bool = False,
    ) -> None:
        super().__init__()
        self.poolers = torch.nn.ModuleList(
            [
                ROIAlignLite(
                    output_size=output_size,
                    spatial_scale=scale,
                    sampling_ratio=sampling_ratio,
                    aligned=aligned,
                )
                for scale in scales
            ]
        )
        self.output_size = output_size
        self.cat_all_levels = cat_all_levels
        lvl_min = int(-math.log2(float(scales[0])))
        lvl_max = int(-math.log2(float(scales[-1])))
        self.map_levels = LevelMapper(lvl_min, lvl_max)
        if self.cat_all_levels:
            self.reduce_channel = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels * len(self.poolers), in_channels, kernel_size=3, stride=1, padding=1)
            )
            self.reduce_relu = torch.nn.ReLU(inplace=True)

    def convert_to_roi_format(self, boxes: list[BoxListLite]) -> torch.Tensor:
        concat_boxes = _cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = _cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        return torch.cat([ids, concat_boxes], dim=1)

    def forward(self, features: list[torch.Tensor], boxes: list[BoxListLite]) -> torch.Tensor:
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if rois.size(0) == 0:
            raise ValueError("No rois were provided to PoolerLite.")
        if num_levels == 1:
            return self.poolers[0](features[0], rois)

        levels = self.map_levels(boxes)
        num_rois = len(rois)
        num_channels = features[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = features[0].dtype, features[0].device
        final_channels = num_channels * num_levels if self.cat_all_levels else num_channels
        result = torch.zeros((num_rois, final_channels, output_size, output_size), dtype=dtype, device=device)

        for level, (per_level_feature, pooler) in enumerate(zip(features, self.poolers)):
            if self.cat_all_levels:
                result[:, level * num_channels:(level + 1) * num_channels, :, :] = pooler(per_level_feature, rois).to(dtype)
            else:
                idx_in_level = torch.nonzero(levels == level).squeeze(1)
                if idx_in_level.numel() == 0:
                    continue
                rois_per_level = rois[idx_in_level]
                result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)
        if self.cat_all_levels:
            result = self.reduce_relu(self.reduce_channel(result))
        return result
