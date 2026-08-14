from __future__ import annotations

import torch
from torch import nn

from services.scene_graph_service.app.layers.make_layers_lite import make_fc
from services.scene_graph_service.app.modules.box_feature_extractors_lite import FPN2MLPFeatureExtractorLite
from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite
from services.scene_graph_service.app.structures.boxlist_ops_lite import boxlist_union_lite


class RelationFeatureExtractorLite(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        pooler_resolution: int,
        pooler_scales: tuple[float, ...] | list[float],
        pooler_sampling_ratio: int,
        mlp_head_dim: int,
        pool_all_levels: bool = True,
        separate_spatial: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = FPN2MLPFeatureExtractorLite(
            in_channels=in_channels,
            pooler_resolution=pooler_resolution,
            pooler_scales=pooler_scales,
            pooler_sampling_ratio=pooler_sampling_ratio,
            mlp_head_dim=mlp_head_dim,
            cat_all_levels=pool_all_levels,
        )
        self.out_channels = self.feature_extractor.out_channels
        self.separate_spatial = separate_spatial
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(
                make_fc(input_size, out_dim // 2),
                nn.ReLU(inplace=True),
                make_fc(out_dim // 2, out_dim),
                nn.ReLU(inplace=True),
            )
        self.rect_size = pooler_resolution * 4 - 1
        self.rect_conv = nn.Sequential(
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        proposals: list[BoxListLite],
        rel_pair_idxs: list[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        device = features[0].device
        union_proposals: list[BoxListLite] = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposals.append(boxlist_union_lite(head_proposal, tail_proposal))

            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            head_rect = (
                (dummy_x_range >= head_proposal.bbox[:, 0].floor().view(-1, 1, 1).long())
                & (dummy_x_range <= head_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long())
                & (dummy_y_range >= head_proposal.bbox[:, 1].floor().view(-1, 1, 1).long())
                & (dummy_y_range <= head_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())
            ).float()
            tail_rect = (
                (dummy_x_range >= tail_proposal.bbox[:, 0].floor().view(-1, 1, 1).long())
                & (dummy_x_range <= tail_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long())
                & (dummy_y_range >= tail_proposal.bbox[:, 1].floor().view(-1, 1, 1).long())
                & (dummy_y_range <= tail_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())
            ).float()
            rect_inputs.append(torch.stack((head_rect, tail_rect), dim=1))

        rect_inputs_tensor = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs_tensor)
        union_vis_features = self.feature_extractor.pooler(features, union_proposals)
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            return region_features, spatial_features
        union_features = union_vis_features + rect_features
        return self.feature_extractor.forward_without_pool(union_features)
