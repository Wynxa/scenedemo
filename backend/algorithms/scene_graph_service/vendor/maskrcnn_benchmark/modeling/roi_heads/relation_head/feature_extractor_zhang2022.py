import torch
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
)
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("Zhang2022RelationFeatureExtractor")
class Zhang2022RelationFeatureExtractor(nn.Module):
    """Paper-style relation extractor with single-level and multi-level union visual features."""

    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg.clone()
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS

        if cfg.MODEL.ATTRIBUTE_ON:
            raise NotImplementedError("Zhang2022RelationFeatureExtractor currently supports ATTRIBUTE_ON=False only.")

        self.feature_extractor = make_roi_box_feature_extractor(
            cfg, in_channels, cat_all_levels=pool_all_levels
        )
        self.out_channels = self.feature_extractor.out_channels

    def forward(self, x, proposals, rel_pair_idxs=None):
        union_proposals = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposals.append(boxlist_union(head_proposal, tail_proposal))

        if len(union_proposals) == 0:
            empty = x[0].new_zeros((0, self.feature_extractor.out_channels))
            return empty, empty

        # e_ij^vis: the standard union RoI feature routed by the FPN level mapper.
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        region_features = self.feature_extractor.forward_without_pool(union_vis_features)

        # b_ij in Eq.12: pool the same union RoI from every FPN level and sum them.
        pooler = self.feature_extractor.pooler
        rois = pooler.convert_to_roi_format(union_proposals)
        multi_level_sum = None
        for per_level_feature, roi_pooler in zip(x, pooler.poolers):
            pooled = roi_pooler(per_level_feature, rois)
            multi_level_sum = pooled if multi_level_sum is None else (multi_level_sum + pooled)
        multi_level_features = self.feature_extractor.forward_without_pool(multi_level_sum)
        return region_features, multi_level_features
