# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import os

import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext, TransformerEncoder_HTCL
from .predictor_zhang2022 import Zhang2022Predictor  # noqa: F401
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 

from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.config import cfg
from .loss import SupConLoss

@registry.ROI_RELATION_PREDICTOR.register("PENET_HTCL")
class PENET_HTCL(nn.Module):
    def __init__(self, config, in_channels):
        super(PENET_HTCL, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048  # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300  # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2  # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                          wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
                                     wv_dim=self.embed_dim)  # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

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

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        fine_cls_layer = self.cfg.MODEL.refine_layers
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD  # 8
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM  # 64
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  # 64
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM  # 2048
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 0.1
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.last_dim = self.cfg.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM
        self.encoder_q = TransformerEncoder_HTCL(fine_cls_layer, self.num_head, self.k_dim, self.v_dim,
                                                 self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.Middle_FEATS_DIM = config.MODEL.ROI_RELATION_HEAD.Middle_FEATS_DIM

        self.embed_dim2 = self.cfg.MODEL.semantic_embed_dim
        rel_embed_vecs2 = rel_vectors(rel_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim2)
        obj_embed_vecs2 = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim2)
        self.rel_embed2 = nn.Embedding(self.num_rel_cls, self.embed_dim2)  # 51 -> 200
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim2)  # 151 -> 200
        self.sub_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim2)  # 151 -> 200

        self.rel_fine_compress = nn.Linear(self.Middle_FEATS_DIM + self.embed_dim2*3, self.hidden_dim)
        with torch.no_grad():
            self.rel_embed2.weight.copy_(rel_embed_vecs2, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs2, non_blocking=True)
            self.sub_embed2.weight.copy_(obj_embed_vecs2, non_blocking=True)

        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.ce_loss = nn.CrossEntropyLoss()
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.ctx_compress_final = nn.Linear(self.hidden_dim, self.last_dim)
        layer_init(self.ctx_compress_final, xavier=True)
        self.fine_cls = nn.Linear(self.last_dim, self.num_rel_cls)
        layer_init(self.fine_cls, xavier=True)

        self.mask_rate = self.cfg.MODEL.refine_mask_rate

        temperature = 0.1
        contrast_mode = 'all'
        base_temperature = 0.07
        self.SupConLoss = SupConLoss(temperature, contrast_mode, base_temperature)
        self.supcon_proj = nn.Sequential(
            nn.Linear(self.last_dim, self.last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_dim, 128)
        )
        layer_init(self.supcon_proj[0], xavier=True)
        layer_init(self.supcon_proj[2], xavier=True)

        self.register_buffer("iter", torch.zeros(1, dtype=torch.long))

        if self.num_rel_cls == len(rel_num):
            if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                rels_number = rel_num_sggen
            else:
                rels_number = rel_num
        else:
            fg_matrix = statistics['fg_matrix']
            rels_number = fg_matrix.sum((0, 1)).float()
            if rels_number.numel() != self.num_rel_cls:
                raise ValueError(
                    "Predicate frequency vector shape {} does not match NUM_CLASSES {}".format(
                        tuple(rels_number.shape), self.num_rel_cls
                    )
                )
            rels_number = rels_number.to(obj_embed_vecs.device)
            rels_number[rels_number <= 0] = 1.0

        sample_num = rels_number.clone()
        sample_num[sample_num < cfg.MODEL.cut_rels] = cfg.MODEL.cut_rels
        relation_device = rels_number.device

        if cfg.MODEL.PENET_wt:
            self.final_cls_weight = torch.FloatTensor([0.0418, 0.1109, 2.1169, 1.3740, 1.1754, 1.2242, 0.4437, 0.4318,
                                                       0.1235, 1.2934, 1.2475, 0.8236, 1.8305, 1.1202, 0.9135, 2.1414,
                                                       0.4787, 1.4416, 2.2787, 0.4441, 0.0446, 0.1203, 0.0455, 0.3264,
                                                       1.1399, 1.3780, 2.2594, 1.7788, 1.8779, 0.0675, 0.0544, 0.0419,
                                                       2.2067, 0.4703, 1.3727, 1.5585, 2.0469, 1.1191, 0.4936, 1.8878,
                                                       0.2460, 0.3163, 1.6831, 0.2068, 2.1942, 2.4253, 0.9280, 1.2198,
                                                       0.0563, 0.2921, 0.0862]).to(relation_device)
        else:
            self.final_cls_weight = torch.FloatTensor(weight_calculate(self.cfg.MODEL.num_beta, sample_num)).to(relation_device)

        self.feature_generation = config.MODEL.Feature_Generation_Mode


        self.log_fre_wt = (torch.log(rels_number) - torch.log(rels_number)[1:].mean()).to(relation_device)
        self.logit_wt = nn.Parameter(self.log_fre_wt)
        self.sigmoid = nn.Sigmoid()

        self.bg_gate_enabled = self.cfg.MODEL.BG_GATE.ENABLED
        self.bg_gate_loss_weight = self.cfg.MODEL.BG_GATE.LOSS_WEIGHT
        self.bg_gate_infer_thres = self.cfg.MODEL.BG_GATE.INFER_THRES
        self.bg_gate_margin = self.cfg.MODEL.BG_GATE.MARGIN
        self.bg_gate_detach_coarse = self.cfg.MODEL.BG_GATE.DETACH_COARSE
        self.bg_gate_apply_to_htcl_only = self.cfg.MODEL.BG_GATE.APPLY_TO_HTCL_ONLY
        if self.bg_gate_enabled:
            bg_gate_hidden = self.cfg.MODEL.BG_GATE.HIDDEN_DIM
            self.bg_gate_mlp = nn.Sequential(
                nn.Linear(5, bg_gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(bg_gate_hidden, 1)
            )
            layer_init(self.bg_gate_mlp[0], xavier=True)
            layer_init(self.bg_gate_mlp[2], xavier=True)

        self.conflict_reentry_enabled = self.cfg.MODEL.CONFLICT_REENTRY.ENABLED
        self.latest_tb_stats = {}
        self.conflict_recall_threshold = 1.0
        self.conflict_mlp_loss_weight = 0.0
        self.conflict_distill_loss_weight = 0.0
        self.conflict_distill_temperature = 1.0
        self.conflict_prior_temperature = 1.0
        self.conflict_train_use_bg_recalled_only = False
        self.conflict_bg_suppress_beta = 0.0
        self.conflict_groups = []
        self.conflict_group_heads = []
        self.conflict_group_lookup = torch.full((self.num_rel_cls,), -1, dtype=torch.long)
        self.conflict_max_group_size = 0
        self.conflict_declared_num_groups = 0
        if self.conflict_reentry_enabled:
            self.conflict_recall_threshold = self.cfg.MODEL.CONFLICT_REENTRY.RECALL_THRESHOLD
            self.conflict_mlp_loss_weight = self.cfg.MODEL.CONFLICT_REENTRY.MLP_LOSS_WEIGHT
            self.conflict_distill_loss_weight = self.cfg.MODEL.CONFLICT_REENTRY.DISTILL_LOSS_WEIGHT
            self.conflict_distill_temperature = self.cfg.MODEL.CONFLICT_REENTRY.DISTILL_TEMPERATURE
            self.conflict_prior_temperature = self.cfg.MODEL.CONFLICT_REENTRY.PRIOR_TEMPERATURE
            self.conflict_train_use_bg_recalled_only = (
                self.cfg.MODEL.CONFLICT_REENTRY.TRAIN_USE_BG_RECALLED_ONLY
            )
            self.conflict_bg_suppress_beta = self.cfg.MODEL.CONFLICT_REENTRY.BG_SUPPRESS_BETA
            self.conflict_rel_proj = nn.Sequential(
                nn.Linear(
                    self.mlp_dim * 2,
                    self.cfg.MODEL.CONFLICT_REENTRY.REL_HIDDEN_DIM,
                ),
                nn.LayerNorm(self.cfg.MODEL.CONFLICT_REENTRY.REL_HIDDEN_DIM),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(
                    self.cfg.MODEL.CONFLICT_REENTRY.REL_HIDDEN_DIM,
                    self.cfg.MODEL.CONFLICT_REENTRY.REL_PROJ_DIM,
                ),
            )
            layer_init(self.conflict_rel_proj[0], xavier=True)
            layer_init(self.conflict_rel_proj[3], xavier=True)
            self.conflict_class_embed = nn.Embedding(
                self.num_rel_cls,
                self.cfg.MODEL.CONFLICT_REENTRY.STAT_CLASS_EMBED_DIM,
            )
            self.conflict_candidate_mlp = nn.Sequential(
                nn.Linear(
                    9 + 30 + self.cfg.MODEL.CONFLICT_REENTRY.STAT_CLASS_EMBED_DIM
                    + self.cfg.MODEL.CONFLICT_REENTRY.REL_PROJ_DIM,
                    self.cfg.MODEL.CONFLICT_REENTRY.MLP_HIDDEN_DIM,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg.MODEL.CONFLICT_REENTRY.MLP_HIDDEN_DIM, 1),
            )
            layer_init(self.conflict_candidate_mlp[0], xavier=True)
            layer_init(self.conflict_candidate_mlp[2], xavier=True)
            (
                self.conflict_groups,
                self.conflict_group_heads,
                self.conflict_group_lookup,
                self.conflict_max_group_size,
            ) = self._load_conflict_groups(
                self.cfg.MODEL.CONFLICT_REENTRY.GROUPS_JSON
            )
            alpha_init = [self.cfg.MODEL.CONFLICT_REENTRY.ALPHA_BASE] * len(self.conflict_groups)
            cfg_alpha_init = list(self.cfg.MODEL.CONFLICT_REENTRY.ALPHA_INIT)
            for idx, value in enumerate(cfg_alpha_init[: len(alpha_init)]):
                alpha_init[idx] = float(value)
            alpha_tensor = torch.tensor(alpha_init, dtype=torch.float32)
            self.group_alpha = nn.Parameter(
                alpha_tensor,
                requires_grad=self.cfg.MODEL.CONFLICT_REENTRY.ALPHA_LEARNABLE,
            )

    def _load_conflict_groups(self, group_json_path):
        groups = []
        group_heads = []
        group_lookup = torch.full((self.num_rel_cls,), -1, dtype=torch.long)
        max_group_size = 0
        if not group_json_path or not os.path.exists(group_json_path):
            return groups, group_heads, group_lookup, max_group_size

        with open(group_json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

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
        non_group_mask = torch.ones(
            self.num_rel_cls,
            device=rel_logits_rows.device,
            dtype=torch.bool,
        )
        non_group_mask[0] = False
        non_group_mask[group_tensor] = False
        if bool(non_group_mask[1:].any().item()):
            z_max_non_group = rel_logits_rows[:, non_group_mask].max(dim=1, keepdim=True).values
        else:
            z_max_non_group = rel_logits_rows.new_zeros(batch_size, 1)

        feat_num = torch.empty(
            batch_size, group_size, 9, device=rel_logits_rows.device, dtype=rel_logits_rows.dtype
        )
        feat_num[:, :, 0] = z_group - z_h
        feat_num[:, :, 1] = z_h - z_group
        feat_num[:, :, 2] = pi_group
        head_local = int((group_tensor == head_cls).nonzero(as_tuple=False)[0].item())
        p_head = pi_group[:, head_local].unsqueeze(1)
        feat_num[:, :, 3] = p_head.expand(-1, group_size)
        feat_num[:, :, 4] = p_head - pi_group
        feat_num[:, :, 5] = rel_logits_rows[:, 0].unsqueeze(1) - z_group
        feat_num[:, :, 6] = rel_logits_rows[:, 0].unsqueeze(1) - z_h
        feat_num[:, :, 7] = z_group - z_max_non_group
        feat_num[:, :, 8] = z_h - z_max_non_group

        group_proto = predicate_proto_norm[group_tensor]
        head_proto = predicate_proto_norm[head_cls]
        rel_vec = rel_rep_norm_rows.detach()
        m_group = rel_vec.unsqueeze(1) * group_proto.unsqueeze(0)
        m_head = (
            rel_vec.unsqueeze(1) * head_proto.view(1, 1, -1)
        ).expand(batch_size, group_size, -1)
        delta_group = m_group - m_head
        flat_shape = batch_size * group_size
        match_stats = torch.cat(
            [
                self._build_match_stats(m_group.reshape(flat_shape, -1)).view(batch_size, group_size, -1),
                self._build_match_stats(m_head.reshape(flat_shape, -1)).view(batch_size, group_size, -1),
                self._build_match_stats(delta_group.reshape(flat_shape, -1)).view(batch_size, group_size, -1),
            ],
            dim=2,
        )
        class_embed = self.conflict_class_embed(group_tensor).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([feat_num, match_stats, class_embed], dim=2)

    def _compute_conflict_distill_loss(
        self,
        mlp_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        routed_group_ids: torch.Tensor,
        distill_mask: torch.Tensor,
    ) -> torch.Tensor:
        total_weight = 0
        total_loss = mlp_logits.new_tensor(0.0)
        temperature = self.conflict_distill_temperature
        for group_id, group_member_ids in enumerate(self.conflict_groups):
            group_sample_mask = distill_mask & (routed_group_ids == group_id)
            if not bool(group_sample_mask.any().item()):
                continue
            group_indices = torch.nonzero(group_sample_mask, as_tuple=False).flatten()
            group_tensor = torch.tensor(group_member_ids, device=mlp_logits.device, dtype=torch.long)
            student_logits = mlp_logits[group_indices][:, group_tensor]
            teacher_group_logits = teacher_logits[group_indices][:, group_tensor].detach()
            teacher_prob = F.softmax(teacher_group_logits / temperature, dim=1)
            student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
            batch_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
            total_loss = total_loss + batch_loss * group_indices.numel()
            total_weight += int(group_indices.numel())
        if total_weight == 0:
            return mlp_logits.new_tensor(0.0)
        return total_loss / float(total_weight)

    def _run_conflict_reentry(
        self,
        rel_logits: torch.Tensor,
        rel_rep: torch.Tensor,
        rel_rep_norm: torch.Tensor,
        predicate_proto_norm: torch.Tensor,
        bg_gate_prob: torch.Tensor,
        recalled_mask: torch.Tensor,
        rel_labels_cat: torch.Tensor = None,
    ):
        device = rel_logits.device
        entry_logits = rel_logits.new_zeros(rel_logits.size(0), self.num_rel_cls)
        entry_prior = rel_logits.new_zeros(rel_logits.size(0), self.num_rel_cls)
        stats = {
            "bg_recall_rate": rel_logits.new_tensor(0.0),
            "recalled_conflict_rate": rel_logits.new_tensor(0.0),
            "recalled_nonconflict_rate": rel_logits.new_tensor(0.0),
            "routing_hit_rate": rel_logits.new_tensor(0.0),
            "mlp_path_rate": rel_logits.new_tensor(0.0),
            "distill_path_rate": rel_logits.new_tensor(0.0),
            "mlp_ce_loss": rel_logits.new_tensor(0.0),
            "mlp_distill_loss": rel_logits.new_tensor(0.0),
            "mlp_group_top1_acc": rel_logits.new_tensor(0.0),
            "entry_prior_mass": rel_logits.new_tensor(0.0),
            "entry_prior_peak_mean": rel_logits.new_tensor(0.0),
            "conflict_loaded_num_groups": rel_logits.new_tensor(float(len(self.conflict_groups))),
            "conflict_declared_num_groups": rel_logits.new_tensor(float(getattr(self, "conflict_declared_num_groups", len(self.conflict_groups)))),
        }
        if (not self.conflict_reentry_enabled) or len(self.conflict_groups) == 0:
            return entry_prior, entry_logits, rel_logits.new_tensor(0.0), stats, {
                "routed_group_ids": rel_logits.new_full((rel_logits.size(0),), -1, dtype=torch.long),
                "mlp_mask": rel_logits.new_zeros(rel_logits.size(0), dtype=torch.bool),
                "distill_mask": rel_logits.new_zeros(rel_logits.size(0), dtype=torch.bool),
            }

        nonbg_top1_cls = rel_logits[:, 1:].argmax(dim=1) + 1
        group_lookup = self.conflict_group_lookup.to(device=device)
        routed_group_ids = group_lookup[nonbg_top1_cls]
        conflict_mask = routed_group_ids >= 0
        mlp_mask = conflict_mask & recalled_mask if self.conflict_train_use_bg_recalled_only else conflict_mask

        stats["bg_recall_rate"] = recalled_mask.float().mean()
        stats["recalled_conflict_rate"] = (recalled_mask & conflict_mask).float().mean()
        stats["recalled_nonconflict_rate"] = (recalled_mask & (~conflict_mask)).float().mean()
        stats["mlp_path_rate"] = mlp_mask.float().mean()

        if rel_labels_cat is not None:
            gt_groups = group_lookup[rel_labels_cat.clamp(min=0, max=self.num_rel_cls - 1)]
            valid_route = recalled_mask & conflict_mask & (rel_labels_cat > 0)
            if bool(valid_route.any().item()):
                stats["routing_hit_rate"] = (
                    gt_groups[valid_route] == routed_group_ids[valid_route]
                ).float().mean()

        total_ce_loss = rel_logits.new_tensor(0.0)
        total_ce_count = 0
        total_top1_correct = 0
        total_top1_count = 0
        distill_mask = rel_logits.new_zeros(rel_logits.size(0), dtype=torch.bool)
        rel_proj = self.conflict_rel_proj(rel_rep)
        for group_id, group_member_ids in enumerate(self.conflict_groups):
            group_sample_mask = mlp_mask & (routed_group_ids == group_id)
            if not bool(group_sample_mask.any().item()):
                continue
            group_indices = torch.nonzero(group_sample_mask, as_tuple=False).flatten()
            group_tensor = torch.tensor(group_member_ids, device=device, dtype=torch.long)
            head_cls = self.conflict_group_heads[group_id]
            stat_feats = self._build_conflict_stat_features_batch(
                rel_logits[group_indices],
                group_tensor,
                head_cls,
                rel_rep_norm[group_indices],
                predicate_proto_norm,
            )
            rel_feat = rel_proj[group_indices].unsqueeze(1).expand(-1, group_tensor.numel(), -1)
            mlp_in = torch.cat([stat_feats, rel_feat], dim=2)
            local_logits = self.conflict_candidate_mlp(
                mlp_in.reshape(-1, mlp_in.size(-1))
            ).view(group_indices.numel(), group_tensor.numel())
            entry_logits[group_indices.unsqueeze(1), group_tensor.unsqueeze(0)] = local_logits
            local_prior = F.softmax(local_logits / self.conflict_prior_temperature, dim=1)
            entry_prior[group_indices.unsqueeze(1), group_tensor.unsqueeze(0)] = local_prior

            if rel_labels_cat is not None:
                group_gt = rel_labels_cat[group_indices]
                target_match = group_gt.unsqueeze(1) == group_tensor.unsqueeze(0)
                valid_target = target_match.any(dim=1)
                if bool(valid_target.any().item()):
                    local_target = target_match[valid_target].float().argmax(dim=1)
                    distill_mask[group_indices[valid_target]] = True
                    batch_ce = F.cross_entropy(local_logits[valid_target], local_target)
                    valid_count = int(valid_target.sum().item())
                    total_ce_loss = total_ce_loss + batch_ce * valid_count
                    total_ce_count += valid_count
                    batch_pred = local_logits[valid_target].argmax(dim=1)
                    total_top1_correct += int((batch_pred == local_target).sum().item())
                    total_top1_count += valid_count

        if total_ce_count > 0:
            mlp_ce_loss = total_ce_loss / float(total_ce_count)
        else:
            mlp_ce_loss = rel_logits.new_tensor(0.0)
        stats["mlp_ce_loss"] = mlp_ce_loss.detach()
        if total_top1_count > 0:
            stats["mlp_group_top1_acc"] = rel_logits.new_tensor(
                float(total_top1_correct) / float(total_top1_count)
            )
        if bool(mlp_mask.any().item()):
            stats["entry_prior_mass"] = entry_prior[mlp_mask].sum(dim=1).mean().detach()
            stats["entry_prior_peak_mean"] = entry_prior[mlp_mask].max(dim=1).values.mean().detach()
        if bool(distill_mask.any().item()):
            stats["distill_path_rate"] = distill_mask.float().mean()
        return entry_prior, entry_logits, mlp_ce_loss, stats, {
            "routed_group_ids": routed_group_ids,
            "mlp_mask": mlp_mask,
            "distill_mask": distill_mask,
        }


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        #####

        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        entity_embeds = self.obj_embed(entity_preds)  # obtaining the word embedding of entities with GloVe

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                                   entity_preds, entity_embeds,
                                                                                   proposals):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists_0 = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        if self.use_bias:
            freq_dist = self.freq_bias.index_with_labels(pair_pred.long())
            rel_dists_0 = rel_dists_0 + freq_dist

        add_losses = {}

        rel_logits = rel_dists_0.clamp(min=1e-5)
        rel_prop = F.softmax(rel_logits, dim=1)
        bg_gate_prob = None
        bg_gate_weight = None
        if self.bg_gate_enabled:
            rel_logits_for_gate = rel_logits.detach() if self.bg_gate_detach_coarse else rel_logits
            bg_logits = rel_logits_for_gate[:, 0]
            nonbg_logits = rel_logits_for_gate[:, 1:]
            nonbg_top1_logits, _ = nonbg_logits.max(dim=1)
            rel_prob_for_gate = F.softmax(rel_logits_for_gate, dim=1)
            bg_prob = rel_prob_for_gate[:, 0]
            nonbg_top1_prob, _ = rel_prob_for_gate[:, 1:].max(dim=1)
            bg_gate_feats = torch.stack(
                [
                    bg_logits,
                    nonbg_top1_logits,
                    bg_logits - nonbg_top1_logits,
                    bg_prob,
                    nonbg_top1_prob,
                ],
                dim=1,
            )
            bg_gate_prob = torch.sigmoid(self.bg_gate_mlp(bg_gate_feats)).squeeze(1)
            if self.training:
                bg_gate_weight = bg_gate_prob
            else:
                bg_gate_weight = (bg_gate_prob >= self.bg_gate_infer_thres).float()
        else:
            bg_gate_prob = rel_logits.new_zeros(rel_logits.size(0))
            bg_gate_weight = rel_logits.new_zeros(rel_logits.size(0))

        rel_labels_cat = cat(rel_labels, dim=0).long() if rel_labels is not None else None
        if self.training and rel_labels_cat is not None:
            coarse_for_target = rel_logits.detach() if self.bg_gate_detach_coarse else rel_logits
            bg_target = (
                (rel_labels_cat != 0)
                & (
                    coarse_for_target[:, 0]
                    > coarse_for_target.gather(1, rel_labels_cat.unsqueeze(1)).squeeze(1)
                    - self.bg_gate_margin
                )
            )
            reentry_recalled_mask = bg_target.bool()
        elif self.conflict_reentry_enabled:
            reentry_recalled_mask = bg_gate_prob >= self.conflict_recall_threshold
        else:
            reentry_recalled_mask = torch.zeros_like(bg_gate_prob, dtype=torch.bool)
        conflict_prior, conflict_logits, mlp_ce_loss, reentry_stats, reentry_cache = self._run_conflict_reentry(
            rel_logits=rel_logits,
            rel_rep=rel_rep,
            rel_rep_norm=rel_rep_norm,
            predicate_proto_norm=predicate_proto_norm,
            bg_gate_prob=bg_gate_prob,
            recalled_mask=reentry_recalled_mask,
            rel_labels_cat=rel_labels_cat,
        )
        rel_prop_entry = rel_prop
        if self.conflict_reentry_enabled and len(self.conflict_groups) > 0:
            routed_group_ids = reentry_cache["routed_group_ids"]
            mlp_mask = reentry_cache["mlp_mask"]
            rel_prop_entry = rel_prop.clone()
            reentry_stats["entry_bg_before_mean"] = rel_prop.new_tensor(0.0)
            reentry_stats["entry_bg_after_mean"] = rel_prop.new_tensor(0.0)
            reentry_stats["entry_fg_after_mean"] = rel_prop.new_tensor(0.0)
            reentry_stats["entry_alpha_mean"] = rel_prop.new_tensor(0.0)
            reentry_stats["entry_alpha_zero_rate"] = rel_prop.new_tensor(0.0)
            reentry_stats["entry_alpha_one_rate"] = rel_prop.new_tensor(0.0)
            if bool(mlp_mask.any().item()):
                alpha_vals = self.group_alpha.to(rel_prop.device)
                alpha_per_sample = rel_prop.new_zeros(rel_prop.size(0))
                alpha_per_sample[mlp_mask] = alpha_vals[routed_group_ids[mlp_mask]]
                alpha_per_sample = alpha_per_sample.clamp(min=0.0, max=1.0)

                reentry_stats["entry_bg_before_mean"] = rel_prop[mlp_mask, 0].mean().detach()
                fused_prop = rel_prop_entry[mlp_mask] * (1.0 - alpha_per_sample[mlp_mask].unsqueeze(1))
                fused_prop = fused_prop + conflict_prior[mlp_mask] * alpha_per_sample[mlp_mask].unsqueeze(1)
                fused_prop = fused_prop / fused_prop.sum(dim=1, keepdim=True).clamp_min(1e-12)
                fused_prop[:, 0] = (1.0 - self.conflict_bg_suppress_beta) * rel_prop[mlp_mask, 0]
                fused_prop = fused_prop / fused_prop.sum(dim=1, keepdim=True).clamp_min(1e-12)
                rel_prop_entry[mlp_mask] = fused_prop
                reentry_stats["entry_alpha_mean"] = alpha_per_sample[mlp_mask].mean().detach()
                reentry_stats["entry_alpha_zero_rate"] = (
                    alpha_per_sample[mlp_mask] <= 1e-6
                ).float().mean().detach()
                reentry_stats["entry_alpha_one_rate"] = (
                    alpha_per_sample[mlp_mask] >= (1.0 - 1e-6)
                ).float().mean().detach()
                reentry_stats["entry_bg_after_mean"] = fused_prop[:, 0].mean().detach()
                reentry_stats["entry_fg_after_mean"] = (1.0 - fused_prop[:, 0]).mean().detach()
        rel_mask = 1 - (rel_prop_entry < 1e-3).long()
        rel_embed_feats = (rel_prop_entry * rel_mask) @ self.rel_embed2.weight

        pair = cat(pair_preds, dim=0)
        sub_feats = self.sub_embed2(pair[:, 0].long())
        obj_feats = self.obj_embed2(pair[:, 1].long())
        to_mask = self.rel_fine_compress(torch.cat([sub_feats, rel_embed_feats, obj_feats, rel_rep], dim=-1))


        if not self.training or self.feature_generation:
            q_feats = self.encoder_q(to_mask, num_rels)
            last_f = self.relu(self.ctx_compress_final(q_feats))
            refine_rel_dists = self.fine_cls(last_f)

            # normalization
            refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1, 1)) / refine_rel_dists.std(dim=1).reshape(-1, 1)
            rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)

            rel_dists_base = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))
            if self.bg_gate_enabled and self.bg_gate_apply_to_htcl_only:
                gate = bg_gate_weight.unsqueeze(1)
                rel_dists_base = rel_dists_0 + gate * (rel_dists_base - rel_dists_0)
            rel_dists = rel_dists_base

            if self.feature_generation:
                rel_labels = cat(rel_labels, dim=0).long()
                return last_f.split(num_rels, dim=0), rel_labels.split(num_rels, dim=0), pair_pred.split(num_rels, dim=0), rel_dists.split(num_rels, dim=0)
            return entity_dists.split(num_objs, dim=0), rel_dists.split(num_rels, dim=0), add_losses

        self.iter[0] += 1

        mean_feats = to_mask.mean(dim=0)
        rate_k = torch.rand(to_mask.shape[0])
        feats_k = to_mask.clone()
        feats_q = to_mask
        feats_k[rate_k < self.mask_rate, :] = mean_feats
        q_k_feats0 = self.encoder_q(cat([feats_q, feats_k]), num_rels + num_rels)

        last_f_q_k = self.relu(self.ctx_compress_final(q_k_feats0))
        last_f, last_k_feats = torch.split(last_f_q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        q_k_rel_dists = self.fine_cls(last_f_q_k)
        refine_rel_dists, mask_rel_dists = torch.split(q_k_rel_dists, [feats_q.shape[0], feats_q.shape[0]], dim=0)

        q_k = self.supcon_proj(last_f_q_k)
        q_k = nn.functional.normalize(q_k, dim=1)
        q0, k0 = torch.split(q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        features = torch.cat([q0.unsqueeze(1), k0.unsqueeze(1)], dim=1)
        selfcon_loss = self.SupConLoss(features)
        add_losses.update(dict(loss_selfcon=selfcon_loss))


        # normalization
        refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1, 1)) / refine_rel_dists.std(dim=1).reshape(-1, 1)
        rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)

        rel_dists_base = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))
        if self.bg_gate_enabled and self.bg_gate_apply_to_htcl_only:
            gate = bg_gate_weight.unsqueeze(1)
            rel_dists_base = rel_dists_0 + gate * (rel_dists_base - rel_dists_0)
        mlp_kd_loss = rel_logits.new_tensor(0.0)
        if self.conflict_reentry_enabled and len(self.conflict_groups) > 0:
            mlp_kd_loss = self._compute_conflict_distill_loss(
                mlp_logits=conflict_logits,
                teacher_logits=rel_dists_base,
                routed_group_ids=reentry_cache["routed_group_ids"],
                distill_mask=reentry_cache["distill_mask"],
            )
        reentry_stats["mlp_distill_loss"] = mlp_kd_loss.detach()
        rel_dists = rel_dists_base

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        loss_coarse = self.ce_loss(rel_dists_0, cat(rel_labels, dim=0))
        add_losses.update(dict(l_dynamic= loss_coarse))

        if cfg.MODEL.reweight_fineloss:
            loss_rel = torch.nn.functional.cross_entropy(cat(rel_dists, dim=0), cat(rel_labels, dim=0),
                                                         self.final_cls_weight)
            add_losses.update(dict(loss_rw= loss_rel))
        if self.bg_gate_enabled:
            coarse_for_target = rel_logits.detach() if self.bg_gate_detach_coarse else rel_logits
            bg_target = ((rel_labels_cat != 0) & (coarse_for_target[:, 0] > coarse_for_target.gather(1, rel_labels_cat.unsqueeze(1)).squeeze(1) - self.bg_gate_margin)).float()
            loss_bg_gate = F.binary_cross_entropy(bg_gate_prob, bg_target)
            add_losses.update(dict(loss_bg_gate=self.bg_gate_loss_weight * loss_bg_gate))
        if self.conflict_reentry_enabled and len(self.conflict_groups) > 0:
            add_losses.update(
                dict(
                    loss_conflict_mlp=self.conflict_mlp_loss_weight * mlp_ce_loss,
                    loss_conflict_distill=self.conflict_distill_loss_weight * mlp_kd_loss,
                )
            )


        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
        simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls * self.num_rel_cls)
        add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end

        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)
        predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(
            dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"dist_loss2": dist_loss})
        ### end

        ###  Prototype-based Learning  ---- Euclidean distance
        rel_labels = cat(rel_labels, dim=0)
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
        predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(
            dim=2) ** 2  # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()
        mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(
            dim=1) / 10  # obtaining g-, where k1 = 10,
        loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(),
                             distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses.update({"loss_dis": loss_sum})  # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end


        if self.cfg.MODEL.ct_loss:
            tb_stats = {
                "bg_gate_prob_mean": bg_gate_prob.detach().mean(),
                "bg_gate_prob_std": bg_gate_prob.detach().std(unbiased=False),
            }
            tb_stats.update({k: v.detach() if torch.is_tensor(v) else v for k, v in reentry_stats.items()})
            if self.conflict_reentry_enabled and len(self.conflict_groups) > 0:
                for group_idx, alpha_val in enumerate(self.group_alpha.detach()):
                    tb_stats[f"alpha_group_{group_idx:02d}"] = alpha_val
                tb_stats["alpha_group_mean"] = self.group_alpha.detach().mean()
            self.latest_tb_stats = tb_stats
            return entity_dists, rel_dists, [add_losses, last_f[rel_labels!=0,:], rel_labels[rel_labels!=0]]

        tb_stats = {
            "bg_gate_prob_mean": bg_gate_prob.detach().mean(),
            "bg_gate_prob_std": bg_gate_prob.detach().std(unbiased=False),
        }
        tb_stats.update({k: v.detach() if torch.is_tensor(v) else v for k, v in reentry_stats.items()})
        if self.conflict_reentry_enabled and len(self.conflict_groups) > 0:
            for group_idx, alpha_val in enumerate(self.group_alpha.detach()):
                tb_stats[f"alpha_group_{group_idx:02d}"] = alpha_val
            tb_stats["alpha_group_mean"] = self.group_alpha.detach().mean()
        self.latest_tb_stats = tb_stats

        return entity_dists, rel_dists, add_losses

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor_HTCL")
class TransformerPredictor_HTCL(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor_HTCL, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON

        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)

        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        layer_init(self.post_cat, xavier=True)

        self.Middle_FEATS_DIM = config.MODEL.ROI_RELATION_HEAD.Middle_FEATS_DIM
        self.rel_compress = nn.Linear(self.pooling_dim + self.hidden_dim * 2, self.Middle_FEATS_DIM)

        self.lrelu = nn.LeakyReLU(0.1)
        layer_init(self.rel_compress, xavier=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.cfg = config

        #### Todo: featute_generation
        self.feature_generation = config.MODEL.Feature_Generation_Mode

        fine_cls_layer = self.cfg.MODEL.refine_layers
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD  # 8
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM  # 64
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  # 64
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM  # 2048
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 0.1
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.last_dim = self.cfg.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM

        self.encoder_q = TransformerEncoder_HTCL(fine_cls_layer, self.num_head, self.k_dim, self.v_dim,
                                                 self.hidden_dim, self.inner_dim, self.dropout_rate)


        self.embed_dim = self.cfg.MODEL.semantic_embed_dim
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)  # 51 -> 200
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 151 -> 200
        self.sub_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 151 -> 200

        self.rel_fine_compress = nn.Linear(self.Middle_FEATS_DIM + self.embed_dim*3, self.hidden_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.sub_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.coarse_cls = nn.Linear(self.Middle_FEATS_DIM, self.num_rel_cls)
        layer_init(self.rel_fine_compress, xavier=True)
        layer_init(self.coarse_cls, xavier=True)

        self.ctx_compress_final = nn.Linear(self.hidden_dim, self.last_dim)
        layer_init(self.ctx_compress_final, xavier=True)
        self.fine_cls = nn.Linear(self.last_dim, self.num_rel_cls)
        layer_init(self.fine_cls, xavier=True)

        self.mask_rate = self.cfg.MODEL.refine_mask_rate

        temperature = 0.1
        contrast_mode = 'all'
        base_temperature = 0.07
        self.SupConLoss = SupConLoss(temperature, contrast_mode, base_temperature)
        self.supcon_proj = nn.Sequential(
            nn.Linear(self.last_dim, self.last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_dim, 128)
        )
        layer_init(self.supcon_proj[0], xavier=True)
        layer_init(self.supcon_proj[2], xavier=True)

        self.register_buffer("iter", torch.zeros(1, dtype=torch.long))

        if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            rels_number = rel_num_sggen
        else:
            rels_number = rel_num

        sample_num = rels_number.clone()
        sample_num[sample_num < cfg.MODEL.cut_rels] = cfg.MODEL.cut_rels

        if cfg.MODEL.PENET_wt:
            self.final_cls_weight = torch.FloatTensor([0.0418, 0.1109, 2.1169, 1.3740, 1.1754, 1.2242, 0.4437, 0.4318,
                                                       0.1235, 1.2934, 1.2475, 0.8236, 1.8305, 1.1202, 0.9135, 2.1414,
                                                       0.4787, 1.4416, 2.2787, 0.4441, 0.0446, 0.1203, 0.0455, 0.3264,
                                                       1.1399, 1.3780, 2.2594, 1.7788, 1.8779, 0.0675, 0.0544, 0.0419,
                                                       2.2067, 0.4703, 1.3727, 1.5585, 2.0469, 1.1191, 0.4936, 1.8878,
                                                       0.2460, 0.3163, 1.6831, 0.2068, 2.1942, 2.4253, 0.9280, 1.2198,
                                                       0.0563, 0.2921, 0.0862]).cuda()
        else:
            self.final_cls_weight = torch.FloatTensor(weight_calculate(self.cfg.MODEL.num_beta, sample_num)).cuda()

        self.log_fre_wt = (torch.log(rels_number) - torch.log(rels_number)[1:].mean()).cuda()
        self.logit_wt = nn.Parameter(self.log_fre_wt)
        self.sigmoid = nn.Sigmoid()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
        visual_rep = ctx_gate * union_features
        middle_feats = self.rel_compress(cat((visual_rep, prod_rep), dim=-1))
        middle_feats = self.lrelu(middle_feats)
        rel_dists_0 = self.coarse_cls(middle_feats)

        # use frequence bias
        if self.use_bias:
            freq_dist = self.freq_bias.index_with_labels(pair_pred.long())
            rel_dists_0 = rel_dists_0 + freq_dist

        add_losses = {}


        rel_logits = rel_dists_0.clamp(min=1e-5)
        rel_prop = F.softmax(rel_logits, dim=1)
        rel_mask = 1 - (rel_prop < 1e-3).long()
        rel_embed_feats = (rel_prop * rel_mask) @ self.rel_embed.weight

        pair = cat(pair_preds, dim=0)
        sub_feats = self.sub_embed(pair[:, 0].long())
        obj_feats = self.obj_embed(pair[:, 1].long())
        to_mask = self.rel_fine_compress(torch.cat([sub_feats, rel_embed_feats, obj_feats, middle_feats], dim=-1))

        if not self.training or self.feature_generation:
            q_feats = self.encoder_q(to_mask, num_rels)
            last_f = self.lrelu(self.ctx_compress_final(q_feats))
            refine_rel_dists = self.fine_cls(last_f)

            # normalization
            refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1,1)) / refine_rel_dists.std(dim=1).reshape(-1, 1)
            rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)
            rel_dists = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))

            if self.feature_generation:
                rel_labels = cat(rel_labels, dim=0).long()
                return last_f.split(num_rels, dim=0), rel_labels.split(num_rels, dim=0), pair_pred.split(num_rels, dim=0), rel_dists_0.split(num_rels, dim=0)
            return obj_dists.split(num_objs, dim=0), rel_dists.split(num_rels, dim=0), add_losses

        self.iter[0] += 1
        rel_labels = cat(rel_labels, dim=0).long()
        mean_feats = to_mask.mean(dim=0)
        rate_k = torch.rand(to_mask.shape[0])
        feats_k = to_mask.clone()
        feats_q = to_mask
        feats_k[rate_k < self.mask_rate, :] = mean_feats
        q_k_feats0 = self.encoder_q(cat([feats_q, feats_k]), num_rels + num_rels)

        last_f_q_k = self.lrelu(self.ctx_compress_final(q_k_feats0))
        last_f, last_k_feats = torch.split(last_f_q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        q_k_rel_dists = self.fine_cls(last_f_q_k)
        refine_rel_dists, mask_rel_dists = torch.split(q_k_rel_dists, [feats_q.shape[0], feats_q.shape[0]], dim=0)

        q_k = self.supcon_proj(last_f_q_k)
        q_k = nn.functional.normalize(q_k, dim=1)
        q0, k0 = torch.split(q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        features = torch.cat([q0.unsqueeze(1), k0.unsqueeze(1)], dim=1)
        selfcon_loss = self.SupConLoss(features)
        add_losses.update(dict(loss_selfcon=selfcon_loss))


        # normalization
        refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1, 1)) / refine_rel_dists.std(dim=1).reshape(-1, 1)
        rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)
        rel_dists = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))

        loss_coarse = self.ce_loss(rel_dists_0, rel_labels)
        add_losses.update(dict(l_dynamic= loss_coarse))

        if cfg.MODEL.reweight_fineloss:
            loss_rel = torch.nn.functional.cross_entropy(rel_dists, rel_labels, self.final_cls_weight)
            add_losses.update(dict(loss_rw= loss_rel))

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        if self.cfg.MODEL.ct_loss:
            return obj_dists, rel_dists, [add_losses, last_f[rel_labels!=0], rel_labels[rel_labels!=0]]

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor_HTCL")
class MotifPredictor_HTCL(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor_HTCL, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        # init contextual lstm encoding
        self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        self.Middle_FEATS_DIM = config.MODEL.ROI_RELATION_HEAD.Middle_FEATS_DIM
        self.rel_compress = nn.Linear(self.pooling_dim, self.Middle_FEATS_DIM)
        self.lrelu = nn.LeakyReLU(0.1)
        layer_init(self.rel_compress, xavier=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.feature_generation = config.MODEL.Feature_Generation_Mode
        fine_cls_layer = self.cfg.MODEL.refine_layers
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD  # 8
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM  # 64
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  # 64
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM  # 2048
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 0.1
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.last_dim = self.cfg.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM

        self.encoder_q = TransformerEncoder_HTCL(fine_cls_layer, self.num_head, self.k_dim, self.v_dim,
                                            self.hidden_dim, self.inner_dim, self.dropout_rate)


        self.embed_dim = self.cfg.MODEL.semantic_embed_dim
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)  # 51 -> 200
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 151 -> 200
        self.sub_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 151 -> 200

        self.rel_fine_compress = nn.Linear(self.Middle_FEATS_DIM + self.embed_dim * 3, self.hidden_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.sub_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.coarse_cls = nn.Linear(self.Middle_FEATS_DIM, self.num_rel_cls)
        layer_init(self.rel_fine_compress, xavier=True)
        layer_init(self.coarse_cls, xavier=True)

        self.ctx_compress_final = nn.Linear(self.hidden_dim, self.last_dim)
        layer_init(self.ctx_compress_final, xavier=True)
        self.fine_cls = nn.Linear(self.last_dim, self.num_rel_cls)
        layer_init(self.fine_cls, xavier=True)

        self.mask_rate = self.cfg.MODEL.refine_mask_rate
        temperature = 0.1
        contrast_mode = 'all'
        base_temperature = 0.07
        self.SupConLoss = SupConLoss(temperature, contrast_mode, base_temperature)
        self.supcon_proj = nn.Sequential(
            nn.Linear(self.last_dim, self.last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_dim, 128)
        )
        layer_init(self.supcon_proj[0], xavier=True)
        layer_init(self.supcon_proj[2], xavier=True)

        self.register_buffer("iter", torch.zeros(1, dtype=torch.long))

        if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            rels_number = rel_num_sggen
        else:
            rels_number = rel_num

        sample_num = rels_number.clone()
        sample_num[sample_num < cfg.MODEL.cut_rels] = cfg.MODEL.cut_rels

        if cfg.MODEL.PENET_wt:
            self.final_cls_weight = torch.FloatTensor([0.0418, 0.1109, 2.1169, 1.3740, 1.1754, 1.2242, 0.4437, 0.4318,
                                                       0.1235, 1.2934, 1.2475, 0.8236, 1.8305, 1.1202, 0.9135, 2.1414,
                                                       0.4787, 1.4416, 2.2787, 0.4441, 0.0446, 0.1203, 0.0455, 0.3264,
                                                       1.1399, 1.3780, 2.2594, 1.7788, 1.8779, 0.0675, 0.0544, 0.0419,
                                                       2.2067, 0.4703, 1.3727, 1.5585, 2.0469, 1.1191, 0.4936, 1.8878,
                                                       0.2460, 0.3163, 1.6831, 0.2068, 2.1942, 2.4253, 0.9280, 1.2198,
                                                       0.0563, 0.2921, 0.0862]).cuda()
        else:
            self.final_cls_weight = torch.FloatTensor(weight_calculate(self.cfg.MODEL.num_beta, sample_num)).cuda()

        self.log_fre_wt = (torch.log(rels_number) - torch.log(rels_number).mean()).cuda()
        self.logit_wt = nn.Parameter(self.log_fre_wt)
        self.sigmoid = nn.Sigmoid()


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        middle_feats = self.rel_compress(prod_rep)
        middle_feats = self.lrelu(middle_feats)

        rel_dists_0 = self.coarse_cls(middle_feats)

        if self.use_bias:
            rel_dists_0 = rel_dists_0 + self.freq_bias.index_with_labels(pair_pred.long())

        rel_logits = rel_dists_0.clamp(min=1e-5)
        rel_prop = F.softmax(rel_logits, dim=1)
        rel_mask = 1 - (rel_prop < 1e-3).long()
        rel_embed_feats = (rel_prop * rel_mask) @ self.rel_embed.weight

        pair = cat(pair_preds, dim=0)
        sub_feats = self.sub_embed(pair[:, 0].long())
        obj_feats = self.obj_embed(pair[:, 1].long())
        to_mask = self.rel_fine_compress(torch.cat([sub_feats, rel_embed_feats, obj_feats, middle_feats], dim=-1))

        add_losses = {}
        if not self.training or self.feature_generation:
            q_feats = self.encoder_q(to_mask, num_rels)

            last_f = self.lrelu(self.ctx_compress_final(q_feats))
            refine_rel_dists = self.fine_cls(last_f)

            # normalization
            refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1,1))/refine_rel_dists.std(dim=1).reshape(-1,1)
            rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)
            rel_dists = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))

            if self.feature_generation:
                rel_labels = cat(rel_labels, dim=0).long()
                return last_f.split(num_rels, dim=0), rel_labels.split(num_rels, dim=0), pair_pred.split(num_rels, dim=0), rel_dists_0.split(num_rels, dim=0)
            return obj_dists.split(num_objs, dim=0), rel_dists.split(num_rels, dim=0), add_losses

        self.iter[0] += 1
        rel_labels = cat(rel_labels, dim=0).long()
        mean_feats = to_mask.mean(dim=0)
        rate_k = torch.rand(to_mask.shape[0])
        feats_k = to_mask.clone()
        feats_q = to_mask
        feats_k[rate_k < self.mask_rate, :] = mean_feats
        q_k_feats0 = self.encoder_q(cat([feats_q, feats_k]), num_rels + num_rels)

        last_f_q_k = self.lrelu(self.ctx_compress_final(q_k_feats0))
        last_f, last_k_feats = torch.split(last_f_q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        q_k_rel_dists = self.fine_cls(last_f_q_k)

        refine_rel_dists, mask_rel_dists = torch.split(q_k_rel_dists, [feats_q.shape[0], feats_q.shape[0]], dim=0)

        q_k = self.supcon_proj(last_f_q_k)
        q_k = nn.functional.normalize(q_k, dim=1)
        q0, k0 = torch.split(q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        features = torch.cat([q0.unsqueeze(1), k0.unsqueeze(1)], dim=1)
        selfcon_loss = self.SupConLoss(features)
        add_losses.update(dict(loss_selfcon=selfcon_loss))


        # normalization
        refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1, 1)) / refine_rel_dists.std(
            dim=1).reshape(-1, 1)
        rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)
        rel_dists = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))

        loss_coarse = self.ce_loss(rel_dists_0, rel_labels)
        add_losses.update(dict(l_dynamic=loss_coarse))

        if cfg.MODEL.reweight_fineloss:
            loss_rel = torch.nn.functional.cross_entropy(rel_dists, rel_labels, self.final_cls_weight)
            add_losses.update(dict(loss_rw=loss_rel))

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        if self.cfg.MODEL.ct_loss:
            return obj_dists, rel_dists, [add_losses, last_f[rel_labels!=0], rel_labels[rel_labels!=0]]

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor_HTCL")
class VCTreePredictor_HTCL(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor_HTCL, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        self.cfg = config
        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        self.Middle_FEATS_DIM = config.MODEL.ROI_RELATION_HEAD.Middle_FEATS_DIM
        self.rel_compress = nn.Linear(self.pooling_dim, self.Middle_FEATS_DIM)

        self.lrelu = nn.LeakyReLU(0.1)
        layer_init(self.rel_compress, xavier=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        self.feature_generation = config.MODEL.Feature_Generation_Mode

        fine_cls_layer = self.cfg.MODEL.refine_layers
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD  # 8
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM  # 64
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  # 64
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM  # 2048
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 0.1
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.last_dim = self.cfg.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM

        self.encoder_q = TransformerEncoder_HTCL(fine_cls_layer, self.num_head, self.k_dim, self.v_dim,
                                            self.hidden_dim, self.inner_dim, self.dropout_rate)


        self.embed_dim = self.cfg.MODEL.semantic_embed_dim
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)  # 51 -> 200
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 151 -> 200
        self.sub_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 151 -> 200

        self.rel_fine_compress = nn.Linear(self.Middle_FEATS_DIM + self.embed_dim * 3, self.hidden_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.sub_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.coarse_cls = nn.Linear(self.Middle_FEATS_DIM, self.num_rel_cls)
        layer_init(self.rel_fine_compress, xavier=True)
        layer_init(self.coarse_cls, xavier=True)

        self.ctx_compress_final = nn.Linear(self.hidden_dim, self.last_dim)
        layer_init(self.ctx_compress_final, xavier=True)
        self.fine_cls = nn.Linear(self.last_dim, self.num_rel_cls)
        layer_init(self.fine_cls, xavier=True)

        self.mask_rate = self.cfg.MODEL.refine_mask_rate

        temperature = 0.1
        contrast_mode = 'all'
        base_temperature = 0.07
        self.SupConLoss = SupConLoss(temperature, contrast_mode, base_temperature)
        self.supcon_proj = nn.Sequential(
            nn.Linear(self.last_dim, self.last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_dim, 128)
        )
        layer_init(self.supcon_proj[0], xavier=True)
        layer_init(self.supcon_proj[2], xavier=True)

        self.register_buffer("iter", torch.zeros(1, dtype=torch.long))

        if self.num_rel_cls == len(rel_num):
            if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                rels_number = rel_num_sggen
            else:
                rels_number = rel_num
        else:
            fg_matrix = statistics['fg_matrix']
            rels_number = fg_matrix.sum((0, 1)).float()
            if rels_number.numel() != self.num_rel_cls:
                raise ValueError(
                    "Predicate frequency vector shape {} does not match NUM_CLASSES {}".format(
                        tuple(rels_number.shape), self.num_rel_cls
                    )
                )
            rels_number = rels_number.to(obj_embed_vecs.device)
            rels_number[rels_number <= 0] = 1.0

        sample_num = rels_number.clone()
        sample_num[sample_num < cfg.MODEL.cut_rels] = cfg.MODEL.cut_rels
        if cfg.MODEL.PENET_wt and self.num_rel_cls == len(rel_num):
            self.final_cls_weight = torch.FloatTensor([0.0418, 0.1109, 2.1169, 1.3740, 1.1754, 1.2242, 0.4437, 0.4318,
                                                       0.1235, 1.2934, 1.2475, 0.8236, 1.8305, 1.1202, 0.9135, 2.1414,
                                                       0.4787, 1.4416, 2.2787, 0.4441, 0.0446, 0.1203, 0.0455, 0.3264,
                                                       1.1399, 1.3780, 2.2594, 1.7788, 1.8779, 0.0675, 0.0544, 0.0419,
                                                       2.2067, 0.4703, 1.3727, 1.5585, 2.0469, 1.1191, 0.4936, 1.8878,
                                                       0.2460, 0.3163, 1.6831, 0.2068, 2.1942, 2.4253, 0.9280, 1.2198,
                                                       0.0563, 0.2921, 0.0862]).cuda()
        else:
            self.final_cls_weight = torch.FloatTensor(weight_calculate(self.cfg.MODEL.num_beta, sample_num)).cuda()

        self.log_fre_wt = (torch.log(rels_number) - torch.log(rels_number)[1:].mean()).cuda()
        self.logit_wt = nn.Parameter(self.log_fre_wt)
        self.sigmoid = nn.Sigmoid()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        middle_feats = self.rel_compress(prod_rep * union_features)
        middle_feats = self.lrelu(middle_feats)

        rel_dists_0 = self.coarse_cls(middle_feats)

        if self.use_bias:
            frq_dists = self.freq_bias.index_with_labels(pair_pred.long())
            rel_dists_0 = rel_dists_0 + frq_dists

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}
        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)



        rel_logits = rel_dists_0.clamp(min=1e-5)
        rel_prop = F.softmax(rel_logits, dim=1)
        rel_mask = 1 - (rel_prop < 1e-3).long()
        rel_embed_feats = (rel_prop * rel_mask) @ self.rel_embed.weight

        pair = cat(pair_preds, dim=0)
        sub_feats = self.sub_embed(pair[:, 0].long())
        obj_feats = self.obj_embed(pair[:, 1].long())
        to_mask = self.rel_fine_compress(torch.cat([sub_feats, rel_embed_feats, obj_feats, middle_feats], dim=-1))


        if not self.training or self.feature_generation:
            q_feats = self.encoder_q(to_mask, num_rels)

            last_f = self.lrelu(self.ctx_compress_final(q_feats))
            refine_rel_dists = self.fine_cls(last_f)

            # normalization
            refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1,1))/refine_rel_dists.std(dim=1).reshape(-1,1)
            rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)
            rel_dists = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))

            if self.feature_generation:
                rel_labels = cat(rel_labels, dim=0).long()
                return last_f.split(num_rels, dim=0), rel_labels.split(num_rels, dim=0), pair_pred.split(num_rels, dim=0), rel_dists_0.split(num_rels, dim=0)
            return obj_dists.split(num_objs, dim=0), rel_dists.split(num_rels, dim=0), add_losses

        self.iter[0] += 1
        rel_labels = cat(rel_labels, dim=0).long()

        mean_feats = to_mask.mean(dim=0)
        rate_k = torch.rand(to_mask.shape[0])
        feats_k = to_mask.clone()
        feats_q = to_mask
        feats_k[rate_k < self.mask_rate, :] = mean_feats
        q_k_feats0 = self.encoder_q(cat([feats_q, feats_k]), num_rels + num_rels)

        last_f_q_k = self.lrelu(self.ctx_compress_final(q_k_feats0))
        last_f, last_k_feats = torch.split(last_f_q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        q_k_rel_dists = self.fine_cls(last_f_q_k)
        refine_rel_dists, mask_rel_dists = torch.split(q_k_rel_dists, [feats_q.shape[0], feats_q.shape[0]], dim=0)

        q_k = self.supcon_proj(last_f_q_k)
        q_k = nn.functional.normalize(q_k, dim=1)
        q0, k0 = torch.split(q_k, [feats_q.shape[0], feats_q.shape[0]], dim=0)
        features = torch.cat([q0.unsqueeze(1), k0.unsqueeze(1)], dim=1)
        selfcon_loss = self.SupConLoss(features)
        add_losses.update(dict(loss_selfcon=selfcon_loss))

        # normalization
        refine_rel_dists = (refine_rel_dists - refine_rel_dists.mean(dim=1).reshape(-1, 1)) / refine_rel_dists.std(dim=1).reshape(-1, 1)
        rel_dists_0 = (rel_dists_0 - rel_dists_0.mean(dim=1).reshape(-1, 1)) / rel_dists_0.std(dim=1).reshape(-1, 1)
        rel_dists = rel_dists_0 * self.sigmoid(self.logit_wt) + refine_rel_dists * (1 - self.sigmoid(self.logit_wt))

        loss_coarse = self.ce_loss(rel_dists_0, rel_labels)
        add_losses.update(dict(l_dynamic=loss_coarse))

        if self.cfg.MODEL.reweight_fineloss:
            loss_rel = torch.nn.functional.cross_entropy(rel_dists, rel_labels, self.final_cls_weight)
            add_losses.update(dict(loss_rw=loss_rel))

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        if self.cfg.MODEL.ct_loss:
            return obj_dists, rel_dists, [add_losses, last_f[rel_labels!=0], rel_labels[rel_labels!=0]]

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("PENET")
class PENET(nn.Module):
    def __init__(self, config, in_channels):
        super(PENET, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048  # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300  # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2  # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                          wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
                                     wv_dim=self.embed_dim)  # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

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

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        if self.num_rel_cls == len(rel_num):
            rels_number = rel_num.to(obj_embed_vecs.device)
        else:
            fg_matrix = statistics['fg_matrix']
            rels_number = fg_matrix.sum((0, 1)).float()
            if rels_number.numel() != self.num_rel_cls:
                raise ValueError(
                    "Predicate frequency vector shape {} does not match NUM_CLASSES {}".format(
                        tuple(rels_number.shape), self.num_rel_cls
                    )
                )
            rels_number = rels_number.to(obj_embed_vecs.device)
            rels_number[rels_number <= 0] = 1.0

        if self.num_rel_cls == len(rel_num):
            self.final_cls_weight = torch.FloatTensor([0.0418, 0.1109, 2.1169, 1.3740, 1.1754, 1.2242, 0.4437, 0.4318,
                                                       0.1235, 1.2934, 1.2475, 0.8236, 1.8305, 1.1202, 0.9135, 2.1414,
                                                       0.4787, 1.4416, 2.2787, 0.4441, 0.0446, 0.1203, 0.0455, 0.3264,
                                                       1.1399, 1.3780, 2.2594, 1.7788, 1.8779, 0.0675, 0.0544, 0.0419,
                                                       2.2067, 0.4703, 1.3727, 1.5585, 2.0469, 1.1191, 0.4936, 1.8878,
                                                       0.2460, 0.3163, 1.6831, 0.2068, 2.1942, 2.4253, 0.9280, 1.2198,
                                                       0.0563, 0.2921, 0.0862]).cuda()
        else:
            self.final_cls_weight = torch.FloatTensor(
                weight_calculate(self.cfg.MODEL.num_beta, rels_number.clone())
            ).cuda()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        #####

        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        entity_embeds = self.obj_embed(entity_preds)  # obtaining the word embedding of entities with GloVe

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                                   entity_preds, entity_embeds,
                                                                                   proposals):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            if cfg.MODEL.reweight_fineloss:
                loss_rel = torch.nn.functional.cross_entropy(cat(rel_dists, dim=0), cat(rel_labels, dim=0),
                                                             self.final_cls_weight)
                add_losses.update(dict(loss_rw=loss_rel))

            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls * self.num_rel_cls)
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end

            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(
                dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(),
                                  -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(
                dim=2) ** 2  # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(
                dim=1) / 10  # obtaining g-, where k1 = 10,
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(),
                                 distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})  # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end

        return entity_dists, rel_dists, add_losses

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x

def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2

def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)


rel_num = torch.tensor([6144612,    5799,     226,     172,     435,     570,    1378, 1178,   11074,     553,     449,
                        1024,     396,     421, 426,       5,     897,     174,     150,     638,   49772, 7293,
                        16957,    3187,     469,     764,     207,      88, 212,   16855,   25549,   85620,     264,
                        1070,     121, 516,     349,      94,    2451,      27,    3428,    1752, 283,    3774,
                        333,     213,     991,     774,   32776, 3454,    9286])

rel_num_sggen = torch.tensor([10306722,   7274,    228,    223,    355,    756,   1786,   1333, 14373,    695,    458,
                              1349,    313,    388,    523,     10, 1029,    154,     97,    588,  65246,  10209,
                              21951,   4004, 652,   1151,    293,     69,    238,  21426,  31832, 102948, 367,   1171,
                              150,    446,    392,    134,   3079,     35, 4288,   2506,    341,   5157,    443,
                              310,   1388,   1240, 55660,   5838,  11732])

def weight_calculate(beta,REL_SAMPLES):
    effective_num = 1.0 - np.power(beta, REL_SAMPLES)
    weights = (1-beta)/np.array(effective_num)
    weights = weights / np.sum(weights) * len(REL_SAMPLES)
    return weights

