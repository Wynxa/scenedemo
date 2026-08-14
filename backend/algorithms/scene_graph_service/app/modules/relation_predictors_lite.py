from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from services.scene_graph_service.app.modules.relation_utils_lite import (
    cat,
    encode_box_info,
    fusion_func,
    layer_init,
    obj_edge_vectors,
    rel_vectors,
    to_onehot,
)
from services.scene_graph_service.app.modules.transformer_lite import TransformerEncoderHTCLLite
from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite


def _get_attr(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def _as_namespace(payload: dict[str, Any]) -> SimpleNamespace:
    converted: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            converted[key] = _as_namespace(value)
        else:
            converted[key] = value
    return SimpleNamespace(**converted)


def build_lite_htcl_cfg(
    *,
    glove_dir: str,
    num_obj_classes: int,
    num_rel_classes: int,
    conflict_groups_json: str = "",
) -> SimpleNamespace:
    payload = {
        "GLOVE_DIR": glove_dir,
        "MODEL": {
            "ROI_BOX_HEAD": {
                "NUM_CLASSES": num_obj_classes,
            },
            "ROI_ATTRIBUTE_HEAD": {
                "NUM_ATTRIBUTES": 1,
            },
            "ROI_RELATION_HEAD": {
                "USE_GT_BOX": True,
                "USE_GT_OBJECT_LABEL": True,
                "NUM_CLASSES": num_rel_classes,
                "CONTEXT_HIDDEN_DIM": 512,
                "CONTEXT_POOLING_DIM": 4096,
                "Middle_FEATS_DIM": 4096,
                "LAST_FEATS_DIM": 4096,
                "PREDICT_USE_VISION": True,
                "PREDICT_USE_BIAS": False,
                "TRANSFORMER": {
                    "NUM_HEAD": 8,
                    "KEY_DIM": 64,
                    "VAL_DIM": 64,
                    "INNER_DIM": 2048,
                    "DROPOUT_RATE": 0.1,
                },
            },
            "num_beta": 0.99995,
            "cut_rels": 200,
            "semantic_embed_dim": 300,
            "reweight_fineloss": False,
            "ct_loss": False,
            "refine_layers": 4,
            "Feature_Generation_Mode": False,
            "refine_mask_rate": 0.0,
            "PENET_wt": False,
            "BG_GATE": {
                "ENABLED": True,
                "HIDDEN_DIM": 64,
                "LOSS_WEIGHT": 0.1,
                "INFER_THRES": 0.5,
                "MARGIN": 0.0,
                "DETACH_COARSE": True,
                "APPLY_TO_HTCL_ONLY": True,
            },
            "CONFLICT_REENTRY": {
                "ENABLED": True,
                "GROUPS_JSON": conflict_groups_json,
                "REL_PROJ_DIM": 64,
                "REL_HIDDEN_DIM": 256,
                "STAT_CLASS_EMBED_DIM": 8,
                "MLP_HIDDEN_DIM": 128,
                "MLP_LOSS_WEIGHT": 0.2,
                "DISTILL_LOSS_WEIGHT": 0.05,
                "DISTILL_TEMPERATURE": 1.0,
                "PRIOR_TEMPERATURE": 1.0,
                "RECALL_THRESHOLD": 0.5,
                "ALPHA_BASE": 0.25,
                "ALPHA_INIT": [],
                "ALPHA_LEARNABLE": True,
                "BG_SUPPRESS_BETA": 0.1,
                "TRAIN_USE_BG_RECALLED_ONLY": True,
            },
        },
        "TEST": {
            "RELATION": {
                "LATER_NMS_PREDICTION_THRES": 0.5,
            },
        },
    }
    return _as_namespace(payload)


@dataclass
class RelationPredictorOutput:
    entity_dists: list[torch.Tensor]
    relation_dists: list[torch.Tensor]
    pair_pred: list[torch.Tensor]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class PENETHTCLLite(nn.Module):
    def __init__(self, cfg: Any, in_channels: int, obj_classes: list[str], rel_classes: list[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.num_att_cls = _get_attr(cfg, "MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES", 1)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.hidden_dim = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM", 512)
        self.pooling_dim = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM", 4096)
        self.mlp_dim = 2048
        self.embed_dim = 300
        self.embed_dim2 = _get_attr(cfg, "MODEL.semantic_embed_dim", 300)
        self.last_dim = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM", 4096)
        self.middle_dim = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.Middle_FEATS_DIM", 4096)
        self.use_bias = bool(_get_attr(cfg, "MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS", False))
        self.use_vision = bool(_get_attr(cfg, "MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION", True))
        self.feature_generation = bool(_get_attr(cfg, "MODEL.Feature_Generation_Mode", False))
        self.mask_rate = float(_get_attr(cfg, "MODEL.refine_mask_rate", 0.0))
        dropout_p = 0.2

        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed2 = nn.Embedding(self.num_rel_cls, self.embed_dim2)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim2)
        self.sub_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim2)

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs2 = rel_vectors(rel_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim2)
        obj_embed_vecs2 = obj_edge_vectors(obj_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim2)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed2.weight.copy_(rel_embed_vecs2, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs2, non_blocking=True)
            self.sub_embed2.weight.copy_(obj_embed_vecs2, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.vis2sem = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(self.mlp_dim * 2, self.mlp_dim),
        )
        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)
        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.lin_obj_cyx = nn.Linear(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        self.num_head = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD", 8)
        self.k_dim = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM", 64)
        self.v_dim = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM", 64)
        self.inner_dim = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM", 2048)
        self.dropout_rate = _get_attr(cfg, "MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE", 0.1)
        fine_cls_layer = int(_get_attr(cfg, "MODEL.refine_layers", 4))
        self.encoder_q = TransformerEncoderHTCLLite(
            fine_cls_layer,
            self.num_head,
            self.k_dim,
            self.v_dim,
            self.hidden_dim,
            self.inner_dim,
            self.dropout_rate,
        )
        self.rel_fine_compress = nn.Linear(self.middle_dim + self.embed_dim2 * 3, self.hidden_dim)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.ctx_compress_final = nn.Linear(self.hidden_dim, self.last_dim)
        self.fine_cls = nn.Linear(self.last_dim, self.num_rel_cls)
        self.supcon_proj = nn.Sequential(
            nn.Linear(self.last_dim, self.last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_dim, 128),
        )
        layer_init(self.ctx_compress_final, xavier=True)
        layer_init(self.fine_cls, xavier=True)
        layer_init(self.supcon_proj[0], xavier=True)
        layer_init(self.supcon_proj[2], xavier=True)
        self.logit_wt = nn.Parameter(torch.zeros(self.num_rel_cls, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

        self.bg_gate_enabled = bool(_get_attr(cfg, "MODEL.BG_GATE.ENABLED", False))
        self.bg_gate_infer_thres = float(_get_attr(cfg, "MODEL.BG_GATE.INFER_THRES", 0.5))
        self.bg_gate_detach_coarse = bool(_get_attr(cfg, "MODEL.BG_GATE.DETACH_COARSE", True))
        self.bg_gate_apply_to_htcl_only = bool(_get_attr(cfg, "MODEL.BG_GATE.APPLY_TO_HTCL_ONLY", True))
        if self.bg_gate_enabled:
            bg_gate_hidden = int(_get_attr(cfg, "MODEL.BG_GATE.HIDDEN_DIM", 64))
            self.bg_gate_mlp = nn.Sequential(
                nn.Linear(5, bg_gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(bg_gate_hidden, 1),
            )
            layer_init(self.bg_gate_mlp[0], xavier=True)
            layer_init(self.bg_gate_mlp[2], xavier=True)

        self.conflict_reentry_enabled = bool(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.ENABLED", False))
        self.conflict_recall_threshold = float(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.RECALL_THRESHOLD", 0.5))
        self.conflict_prior_temperature = float(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.PRIOR_TEMPERATURE", 1.0))
        self.conflict_bg_suppress_beta = float(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.BG_SUPPRESS_BETA", 0.1))
        self.conflict_groups: list[list[int]] = []
        self.conflict_group_heads: list[int] = []
        self.conflict_group_lookup = torch.full((self.num_rel_cls,), -1, dtype=torch.long)
        self.conflict_max_group_size = 0
        self.conflict_declared_num_groups = 0
        if self.conflict_reentry_enabled:
            self.conflict_rel_proj = nn.Sequential(
                nn.Linear(self.mlp_dim * 2, int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.REL_HIDDEN_DIM", 256))),
                nn.LayerNorm(int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.REL_HIDDEN_DIM", 256))),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(
                    int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.REL_HIDDEN_DIM", 256)),
                    int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.REL_PROJ_DIM", 64)),
                ),
            )
            layer_init(self.conflict_rel_proj[0], xavier=True)
            layer_init(self.conflict_rel_proj[3], xavier=True)
            self.conflict_class_embed = nn.Embedding(
                self.num_rel_cls,
                int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.STAT_CLASS_EMBED_DIM", 8)),
            )
            self.conflict_candidate_mlp = nn.Sequential(
                nn.Linear(
                    9 + 30
                    + int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.STAT_CLASS_EMBED_DIM", 8))
                    + int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.REL_PROJ_DIM", 64)),
                    int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.MLP_HIDDEN_DIM", 128)),
                ),
                nn.ReLU(inplace=True),
                nn.Linear(int(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.MLP_HIDDEN_DIM", 128)), 1),
            )
            layer_init(self.conflict_candidate_mlp[0], xavier=True)
            layer_init(self.conflict_candidate_mlp[2], xavier=True)
            (
                self.conflict_groups,
                self.conflict_group_heads,
                self.conflict_group_lookup,
                self.conflict_max_group_size,
            ) = self._load_conflict_groups(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.GROUPS_JSON", ""))
            alpha_init = [float(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.ALPHA_BASE", 0.25))] * len(self.conflict_groups)
            cfg_alpha_init = list(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.ALPHA_INIT", []))
            for idx, value in enumerate(cfg_alpha_init[: len(alpha_init)]):
                alpha_init[idx] = float(value)
            alpha_tensor = torch.tensor(alpha_init, dtype=torch.float32)
            self.group_alpha = nn.Parameter(
                alpha_tensor,
                requires_grad=bool(_get_attr(cfg, "MODEL.CONFLICT_REENTRY.ALPHA_LEARNABLE", True)),
            )

        self.register_buffer("iter", torch.zeros(1, dtype=torch.long))

        use_gt_box = bool(_get_attr(cfg, "MODEL.ROI_RELATION_HEAD.USE_GT_BOX", True))
        use_gt_object_label = bool(_get_attr(cfg, "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL", True))
        if use_gt_box and use_gt_object_label:
            self.mode = "predcls"
        elif use_gt_box:
            self.mode = "sgcls"
        else:
            self.mode = "sgdet"
        if self.mode != "predcls":
            raise NotImplementedError("PENETHTCLLite currently supports predcls only.")

    def _load_conflict_groups(self, group_json_path: str) -> tuple[list[list[int]], list[int], torch.Tensor, int]:
        groups: list[list[int]] = []
        group_heads: list[int] = []
        group_lookup = torch.full((self.num_rel_cls,), -1, dtype=torch.long)
        max_group_size = 0
        if not group_json_path or not os.path.exists(group_json_path):
            return groups, group_heads, group_lookup, max_group_size
        payload = json.loads(open(group_json_path, "r", encoding="utf-8").read())
        declared_num_groups = None
        if isinstance(payload, dict):
            declared_num_groups = payload.get("meta", {}).get("num_groups", None)
        raw_groups = payload.get("conflict_groups", payload if isinstance(payload, list) else [])
        for group_idx, entry in enumerate(raw_groups):
            if not isinstance(entry, dict):
                continue
            member_ids = entry.get("member_ids", [])
            if not member_ids:
                continue
            member_ids = [int(member_id) for member_id in member_ids]
            head_id = int(entry.get("group_head_id", member_ids[0]))
            groups.append(member_ids)
            group_heads.append(head_id)
            max_group_size = max(max_group_size, len(member_ids))
            for member_id in member_ids:
                if 0 <= member_id < self.num_rel_cls:
                    group_lookup[member_id] = group_idx
        self.conflict_declared_num_groups = int(declared_num_groups) if declared_num_groups is not None else len(groups)
        return groups, group_heads, group_lookup, max_group_size

    def _topk_mean(self, values: torch.Tensor, k: int) -> torch.Tensor:
        topk_vals = torch.topk(values, k=min(k, values.size(-1)), dim=-1).values
        return topk_vals.mean(dim=-1)

    def _build_match_stats(self, vec: torch.Tensor) -> torch.Tensor:
        abs_vec = vec.abs()
        pos_mask = vec > 0
        neg_mask = vec < 0
        pos_vals = torch.where(pos_mask, vec, torch.zeros_like(vec))
        topk_pos = self._topk_mean(pos_vals, 3)
        topk_abs = self._topk_mean(abs_vec, 3)
        dim = float(vec.size(-1))
        return torch.stack(
            [
                vec.mean(dim=-1),
                vec.max(dim=-1).values,
                vec.min(dim=-1).values,
                vec.std(dim=-1, unbiased=False),
                abs_vec.sum(dim=-1) / dim,
                vec.norm(p=2, dim=-1) / (dim ** 0.5),
                topk_pos,
                topk_abs,
                pos_mask.float().mean(dim=-1),
                neg_mask.float().mean(dim=-1),
            ],
            dim=-1,
        )

    def _build_conflict_stat_features_batch(
        self,
        rel_logits_rows: torch.Tensor,
        group_tensor: torch.Tensor,
        head_cls: int,
        rel_rep_norm_rows: torch.Tensor,
        predicate_proto_norm: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rel_logits_rows.size(0)
        group_size = group_tensor.numel()
        z_group = rel_logits_rows[:, group_tensor]
        z_h = rel_logits_rows[:, head_cls].unsqueeze(1)
        pi_group = F.softmax(z_group, dim=1)
        non_group_mask = torch.ones(self.num_rel_cls, device=rel_logits_rows.device, dtype=torch.bool)
        non_group_mask[0] = False
        non_group_mask[group_tensor] = False
        if bool(non_group_mask[1:].any().item()):
            z_max_non_group = rel_logits_rows[:, non_group_mask].max(dim=1, keepdim=True).values
        else:
            z_max_non_group = rel_logits_rows.new_zeros(batch_size, 1)

        feat_num = torch.empty(batch_size, group_size, 9, device=rel_logits_rows.device, dtype=rel_logits_rows.dtype)
        feat_num[:, :, 0] = z_group - z_h
        feat_num[:, :, 1] = z_h - z_group
        feat_num[:, :, 2] = pi_group
        head_local = int((group_tensor == head_cls).nonzero(as_tuple=False)[0].item())
        p_head = pi_group[:, head_local].unsqueeze(1)
        feat_num[:, :, 3] = p_head.expand(-1, group_size)
        feat_num[:, :, 4] = p_head - pi_group
        feat_num[:, :, 5] = rel_logits_rows[:, 0].unsqueeze(1) - z_group
        feat_num[:, :, 6] = z_group - z_max_non_group
        feat_num[:, :, 7] = z_h - z_max_non_group
        feat_num[:, :, 8] = rel_logits_rows[:, 0].unsqueeze(1) - z_max_non_group

        group_proto = predicate_proto_norm[group_tensor]
        rel_rep_expand = rel_rep_norm_rows.unsqueeze(1).expand(-1, group_size, -1)
        match_vec = rel_rep_expand - group_proto.unsqueeze(0)
        feat_match = self._build_match_stats(match_vec.reshape(batch_size * group_size, -1)).view(batch_size, group_size, -1)
        proto_sim = torch.matmul(group_proto, group_proto.transpose(0, 1))
        proto_stats = self._build_match_stats(proto_sim).unsqueeze(0).expand(batch_size, -1)
        proto_stats = proto_stats.unsqueeze(1).expand(-1, group_size, -1)
        class_embed = self.conflict_class_embed(group_tensor).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([feat_num, feat_match, proto_stats, class_embed], dim=-1)

    def _run_conflict_reentry(
        self,
        *,
        rel_logits: torch.Tensor,
        rel_rep: torch.Tensor,
        rel_rep_norm: torch.Tensor,
        predicate_proto_norm: torch.Tensor,
        bg_gate_prob: torch.Tensor,
        recalled_mask: torch.Tensor,
        rel_labels_cat: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        stats = {
            "recalled_conflict_rate": rel_logits.new_tensor(0.0),
            "recalled_nonconflict_rate": rel_logits.new_tensor(0.0),
            "conflict_loaded_num_groups": rel_logits.new_tensor(float(len(self.conflict_groups))),
            "conflict_declared_num_groups": rel_logits.new_tensor(float(self.conflict_declared_num_groups)),
        }
        empty_prior = F.softmax(rel_logits, dim=1)
        empty_logits = rel_logits.new_zeros((rel_logits.size(0), self.conflict_max_group_size if self.conflict_max_group_size > 0 else 1))
        empty_cache = {
            "routed_group_ids": rel_logits.new_full((rel_logits.size(0),), -1, dtype=torch.long),
            "mlp_mask": torch.zeros(rel_logits.size(0), dtype=torch.bool, device=rel_logits.device),
            "distill_mask": torch.zeros(rel_logits.size(0), dtype=torch.bool, device=rel_logits.device),
        }
        if (not self.conflict_reentry_enabled) or len(self.conflict_groups) == 0:
            return empty_prior, empty_logits, rel_logits.new_tensor(0.0), stats, empty_cache

        device = rel_logits.device
        rel_prop = F.softmax(rel_logits, dim=1)
        routed_group_ids = self.conflict_group_lookup.to(device=device)[rel_prop[:, 1:].argmax(dim=1) + 1]
        conflict_mask = routed_group_ids >= 0
        mlp_mask = recalled_mask & conflict_mask
        stats["recalled_conflict_rate"] = (recalled_mask & conflict_mask).float().mean()
        stats["recalled_nonconflict_rate"] = (recalled_mask & (~conflict_mask)).float().mean()

        conflict_prior = rel_prop.clone()
        conflict_logits_rows = []
        for group_id, group_member_ids in enumerate(self.conflict_groups):
            row_mask = mlp_mask & (routed_group_ids == group_id)
            if not bool(row_mask.any().item()):
                continue
            group_tensor = torch.tensor(group_member_ids, dtype=torch.long, device=device)
            head_cls = self.conflict_group_heads[group_id]
            stat_feats = self._build_conflict_stat_features_batch(
                rel_logits[row_mask],
                group_tensor,
                head_cls,
                rel_rep_norm[row_mask],
                predicate_proto_norm,
            )
            rel_proj = self.conflict_rel_proj(rel_rep[row_mask]).unsqueeze(1).expand(-1, group_tensor.numel(), -1)
            local_logits = self.conflict_candidate_mlp(torch.cat([stat_feats, rel_proj], dim=-1)).squeeze(-1)
            local_prior = F.softmax(local_logits / self.conflict_prior_temperature, dim=1)
            conflict_prior[row_mask, group_tensor] = local_prior
            conflict_logits_rows.append((row_mask, local_logits))

        max_group = self.conflict_max_group_size if self.conflict_max_group_size > 0 else 1
        conflict_logits = rel_logits.new_zeros((rel_logits.size(0), max_group))
        for row_mask, local_logits in conflict_logits_rows:
            conflict_logits[row_mask, : local_logits.size(1)] = local_logits
        conflict_prior = conflict_prior / conflict_prior.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return conflict_prior, conflict_logits, rel_logits.new_tensor(0.0), stats, {
            "routed_group_ids": routed_group_ids,
            "mlp_mask": mlp_mask,
            "distill_mask": mlp_mask,
        }

    def refine_obj_labels(self, roi_features: torch.Tensor, proposals: list[BoxListLite]) -> tuple[torch.Tensor, torch.Tensor]:
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0).long()
        _ = self.pos_embed(encode_box_info(proposals))
        _ = self.lin_obj_cyx(cat([roi_features, self.obj_embed1(obj_labels), self.pos_embed(encode_box_info(proposals))], -1))
        obj_preds = obj_labels
        obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        return obj_dists, obj_preds

    def forward(
        self,
        proposals: list[BoxListLite],
        rel_pair_idxs: list[torch.Tensor],
        rel_labels: list[torch.Tensor] | None,
        rel_binarys: list[torch.Tensor] | None,
        roi_features: torch.Tensor,
        union_features: torch.Tensor,
        logger: Any | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
        _ = rel_binarys
        _ = logger
        if union_features is None:
            raise ValueError("PENETHTCLLite requires union_features for predcls inference.")

        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        entity_rep = self.post_emb(roi_features).view(roi_features.size(0), 2, self.mlp_dim)
        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)
        entity_embeds = self.obj_embed(entity_preds)

        num_rels = [int(r.shape[0]) for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds_split = entity_preds.split(num_objs, dim=0)
        entity_embeds_split = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []
        for pair_idx, sub_rep_i, obj_rep_i, entity_pred_i, entity_embed_i in zip(
            rel_pair_idxs, sub_reps, obj_reps, entity_preds_split, entity_embeds_split
        ):
            s_embed = self.W_sub(entity_embed_i[pair_idx[:, 0]])
            o_embed = self.W_obj(entity_embed_i[pair_idx[:, 1]])
            sem_sub = self.vis2sem(sub_rep_i[pair_idx[:, 0]])
            sem_obj = self.vis2sem(obj_rep_i[pair_idx[:, 1]])
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))
            sub = s_embed + sem_sub * gate_sem_sub
            obj = o_embed + sem_obj * gate_sem_obj
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            fusion_so.append(fusion_func(sub, obj))
            pair_preds.append(torch.stack((entity_pred_i[pair_idx[:, 0]], entity_pred_i[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        sem_pred = self.vis2sem(self.down_samp(union_features))
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))
        rel_rep = fusion_so - sem_pred * gate_sem_pred
        predicate_proto = self.W_pred(self.rel_embed.weight)

        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True).clamp_min(1e-12)
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rel_dists_0 = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()

        rel_logits = rel_dists_0.clamp(min=1e-5)
        rel_prop = F.softmax(rel_logits, dim=1)
        if self.bg_gate_enabled:
            rel_logits_for_gate = rel_logits.detach() if self.bg_gate_detach_coarse else rel_logits
            bg_logits = rel_logits_for_gate[:, 0]
            nonbg_logits = rel_logits_for_gate[:, 1:]
            nonbg_top1_logits, _ = nonbg_logits.max(dim=1)
            rel_prob_for_gate = F.softmax(rel_logits_for_gate, dim=1)
            bg_prob = rel_prob_for_gate[:, 0]
            nonbg_top1_prob, _ = rel_prob_for_gate[:, 1:].max(dim=1)
            bg_gate_feats = torch.stack(
                [bg_logits, nonbg_top1_logits, bg_logits - nonbg_top1_logits, bg_prob, nonbg_top1_prob],
                dim=1,
            )
            bg_gate_prob = torch.sigmoid(self.bg_gate_mlp(bg_gate_feats)).squeeze(1)
            bg_gate_weight = (bg_gate_prob >= self.bg_gate_infer_thres).float()
        else:
            bg_gate_prob = rel_logits.new_zeros(rel_logits.size(0))
            bg_gate_weight = rel_logits.new_zeros(rel_logits.size(0))

        reentry_recalled_mask = bg_gate_prob >= self.conflict_recall_threshold if self.conflict_reentry_enabled else torch.zeros_like(bg_gate_prob, dtype=torch.bool)
        conflict_prior, conflict_logits, _, _, reentry_cache = self._run_conflict_reentry(
            rel_logits=rel_logits,
            rel_rep=rel_rep,
            rel_rep_norm=rel_rep_norm,
            predicate_proto_norm=predicate_proto_norm,
            bg_gate_prob=bg_gate_prob,
            recalled_mask=reentry_recalled_mask,
            rel_labels_cat=None,
        )
        _ = conflict_logits

        rel_prop_entry = rel_prop.clone()
        if self.conflict_reentry_enabled and len(self.conflict_groups) > 0:
            routed_group_ids = reentry_cache["routed_group_ids"]
            mlp_mask = reentry_cache["mlp_mask"]
            if bool(mlp_mask.any().item()):
                alpha_vals = self.group_alpha.to(rel_prop.device)
                alpha_per_sample = rel_prop.new_zeros(rel_prop.size(0))
                alpha_per_sample[mlp_mask] = alpha_vals[routed_group_ids[mlp_mask]]
                alpha_per_sample = alpha_per_sample.clamp(min=0.0, max=1.0)
                fused_prop = rel_prop_entry[mlp_mask] * (1.0 - alpha_per_sample[mlp_mask].unsqueeze(1))
                fused_prop = fused_prop + conflict_prior[mlp_mask] * alpha_per_sample[mlp_mask].unsqueeze(1)
                fused_prop = fused_prop / fused_prop.sum(dim=1, keepdim=True).clamp_min(1e-12)
                fused_prop[:, 0] = (1.0 - self.conflict_bg_suppress_beta) * rel_prop[mlp_mask, 0]
                fused_prop = fused_prop / fused_prop.sum(dim=1, keepdim=True).clamp_min(1e-12)
                rel_prop_entry[mlp_mask] = fused_prop

        rel_mask = 1 - (rel_prop_entry < 1e-3).long()
        rel_embed_feats = (rel_prop_entry * rel_mask) @ self.rel_embed2.weight
        pair = pair_pred
        sub_feats = self.sub_embed2(pair[:, 0].long())
        obj_feats = self.obj_embed2(pair[:, 1].long())
        to_mask = self.rel_fine_compress(torch.cat([sub_feats, rel_embed_feats, obj_feats, rel_rep], dim=-1))
        q_feats = self.encoder_q(to_mask, num_rels)
        last_f = self.relu(self.ctx_compress_final(q_feats))
        refine_rel_dists = self.fine_cls(last_f)

        refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1, 1)) / refine_rel_dists.std(dim=1).reshape(-1, 1).clamp_min(1e-12)
        rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1).clamp_min(1e-12)
        rel_dists_base = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))
        if self.bg_gate_enabled and self.bg_gate_apply_to_htcl_only:
            gate = bg_gate_weight.unsqueeze(1)
            rel_dists_base = rel_dists_0 + gate * (rel_dists_base - rel_dists_0)
        return entity_dists.split(num_objs, dim=0), rel_dists_base.split(num_rels, dim=0), {}
