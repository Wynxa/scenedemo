from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from services.scene_graph_service.app.modules.box_feature_extractors_lite import FPN2MLPFeatureExtractorLite
from services.scene_graph_service.app.modules.relation_feature_extractors_lite import RelationFeatureExtractorLite
from services.scene_graph_service.app.modules.relation_predictors_lite import PENETHTCLLite
from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite


@dataclass
class RelationHeadLiteOutput:
    obj_logits: list[torch.Tensor]
    rel_logits: list[torch.Tensor]
    rel_pair_idxs: list[torch.Tensor]
    losses: dict[str, Any]


class RelationHeadLite(nn.Module):
    def __init__(
        self,
        *,
        predictor: PENETHTCLLite,
        in_channels: int = 512,
        pooler_resolution: int = 7,
        pooler_scales: tuple[float, ...] = (0.25, 0.125, 0.0625, 0.03125),
        pooler_sampling_ratio: int = 2,
        mlp_head_dim: int = 4096,
        pool_all_levels: bool = True,
    ) -> None:
        super().__init__()
        self.box_feature_extractor = FPN2MLPFeatureExtractorLite(
            in_channels=in_channels,
            pooler_resolution=pooler_resolution,
            pooler_scales=pooler_scales,
            pooler_sampling_ratio=pooler_sampling_ratio,
            mlp_head_dim=mlp_head_dim,
            cat_all_levels=False,
        )
        self.union_feature_extractor = RelationFeatureExtractorLite(
            in_channels=in_channels,
            pooler_resolution=pooler_resolution,
            pooler_scales=pooler_scales,
            pooler_sampling_ratio=pooler_sampling_ratio,
            mlp_head_dim=mlp_head_dim,
            pool_all_levels=pool_all_levels,
            separate_spatial=False,
        )
        self.predictor = predictor

    def forward(
        self,
        *,
        features: list[torch.Tensor],
        proposals: list[BoxListLite],
        rel_pair_idxs: list[torch.Tensor],
        rel_labels: list[torch.Tensor] | None = None,
    ) -> RelationHeadLiteOutput:
        roi_features = self.box_feature_extractor(features, proposals)
        union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        obj_logits, rel_logits, losses = self.predictor(
            proposals,
            rel_pair_idxs,
            rel_labels,
            None,
            roi_features,
            union_features,
        )
        return RelationHeadLiteOutput(
            obj_logits=obj_logits,
            rel_logits=rel_logits,
            rel_pair_idxs=rel_pair_idxs,
            losses=losses,
        )


def load_relation_head_from_checkpoint(
    relation_head: RelationHeadLite,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    box_state_dict = {
        key[len("roi_heads.relation.box_feature_extractor.") :]: value
        for key, value in checkpoint_state_dict.items()
        if key.startswith("roi_heads.relation.box_feature_extractor.")
    }
    union_state_dict = {
        key[len("roi_heads.relation.union_feature_extractor.") :]: value
        for key, value in checkpoint_state_dict.items()
        if key.startswith("roi_heads.relation.union_feature_extractor.")
    }
    predictor_state_dict = {
        key[len("roi_heads.relation.predictor.") :]: value
        for key, value in checkpoint_state_dict.items()
        if key.startswith("roi_heads.relation.predictor.")
    }
    box_result = relation_head.box_feature_extractor.load_state_dict(box_state_dict, strict=False)
    union_result = relation_head.union_feature_extractor.load_state_dict(union_state_dict, strict=False)
    predictor_result = relation_head.predictor.load_state_dict(predictor_state_dict, strict=False)
    return {
        "box_feature_extractor": {
            "missing_keys": list(box_result.missing_keys),
            "unexpected_keys": list(box_result.unexpected_keys),
            "num_checkpoint_keys": len(box_state_dict),
        },
        "union_feature_extractor": {
            "missing_keys": list(union_result.missing_keys),
            "unexpected_keys": list(union_result.unexpected_keys),
            "num_checkpoint_keys": len(union_state_dict),
        },
        "predictor": {
            "missing_keys": list(predictor_result.missing_keys),
            "unexpected_keys": list(predictor_result.unexpected_keys),
            "num_checkpoint_keys": len(predictor_state_dict),
        },
    }
