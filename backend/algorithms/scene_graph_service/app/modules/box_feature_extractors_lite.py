from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from services.scene_graph_service.app.layers.make_layers_lite import make_fc
from services.scene_graph_service.app.ops.pooler_lite import PoolerLite
from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite


class FPN2MLPFeatureExtractorLite(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        pooler_resolution: int,
        pooler_scales: tuple[float, ...] | list[float],
        pooler_sampling_ratio: int,
        mlp_head_dim: int,
        use_gn: bool = False,
        half_out: bool = False,
        cat_all_levels: bool = False,
    ) -> None:
        super().__init__()
        self.pooler = PoolerLite(
            output_size=(pooler_resolution, pooler_resolution),
            scales=tuple(pooler_scales),
            sampling_ratio=pooler_sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * pooler_resolution ** 2
        representation_size = mlp_head_dim
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        out_dim = int(representation_size / 2) if half_out else representation_size
        self.fc7 = make_fc(representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, features: list[torch.Tensor], proposals: list[BoxListLite]) -> torch.Tensor:
        x = self.pooler(features, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

    def forward_without_pool(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x
