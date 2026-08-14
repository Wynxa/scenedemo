import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.utils import cat

from .model_motifs import FrequencyBias
from .model_transformer import TransformerEncoder_HTCL
from .utils_motifs import obj_edge_vectors, to_onehot, encode_box_info
from .utils_relation import layer_init, obj_prediction_nms


class Zhang2022Context(nn.Module):
    def __init__(self, config, obj_classes, in_channels):
        super().__init__()
        self.cfg = config
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = "predcls" if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else "sgcls"
        else:
            self.mode = "sgdet"

        self.obj_classes = obj_classes
        self.num_obj_cls = len(obj_classes)
        self.in_channels = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        trans_cfg = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER
        self.dropout_rate = trans_cfg.DROPOUT_RATE
        self.obj_layer = trans_cfg.OBJ_LAYER
        self.num_head = trans_cfg.NUM_HEAD
        self.inner_dim = trans_cfg.INNER_DIM
        self.k_dim = trans_cfg.KEY_DIM
        self.v_dim = trans_cfg.VAL_DIM

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        self.bbox_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = TransformerEncoder_HTCL(
            self.obj_layer,
            self.num_head,
            self.k_dim,
            self.v_dim,
            self.hidden_dim,
            self.inner_dim,
            self.dropout_rate,
        )

    def forward(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        pos_embed = self.bbox_embed(encode_box_info(proposals))
        obj_pre_rep = self.lin_obj(cat((roi_features, obj_embed, pos_embed), dim=-1))
        num_objs = [len(p) for p in proposals]
        obj_ctx = self.context_obj(obj_pre_rep, num_objs)

        if self.mode == "predcls":
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_ctx)
            use_decoder_nms = self.mode == "sgdet" and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field("boxes_per_cls") for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        obj_embed_ctx = self.obj_embed2(obj_preds.long())
        return obj_dists, obj_preds, obj_ctx, obj_embed_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for cur_dists, cur_boxes in zip(obj_dists, boxes_per_cls):
            obj_preds.append(obj_prediction_nms(cur_boxes, cur_dists, nms_thresh=self.nms_thresh))
        return cat(obj_preds, dim=0)


@registry.ROI_RELATION_PREDICTOR.register("Zhang2022Predictor")
class Zhang2022Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        if self.attribute_on:
            raise NotImplementedError("Zhang2022Predictor currently supports ATTRIBUTE_ON=False only.")

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        self.context_layer = Zhang2022Context(config, obj_classes, in_channels)

        self.union_visual_proj = nn.Linear(self.pooling_dim, self.hidden_dim)
        self.union_spatial_proj = nn.Linear(self.pooling_dim, self.hidden_dim)
        self.pair_semantic = nn.Linear(self.embed_dim * 2, self.hidden_dim)
        self.rel_input_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        trans_cfg = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER
        self.rel_encoder = TransformerEncoder_HTCL(
            trans_cfg.REL_LAYER,
            trans_cfg.NUM_HEAD,
            trans_cfg.KEY_DIM,
            trans_cfg.VAL_DIM,
            self.hidden_dim,
            trans_cfg.INNER_DIM,
            trans_cfg.DROPOUT_RATE,
        )
        self.rel_classifier = nn.Linear(self.hidden_dim, self.num_rel_cls)
        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

        layer_init(self.union_visual_proj, xavier=True)
        layer_init(self.union_spatial_proj, xavier=True)
        layer_init(self.pair_semantic, xavier=True)
        layer_init(self.rel_input_proj, xavier=True)
        layer_init(self.rel_classifier, xavier=True)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, _, obj_embed_ctx = self.context_layer(roi_features, proposals)
        num_objs = [len(p) for p in proposals]
        num_rels = [pair_idx.shape[0] for pair_idx in rel_pair_idxs]

        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_embed_ctxs = obj_embed_ctx.split(num_objs, dim=0)

        if isinstance(union_features, tuple):
            union_vis_splits = union_features[0].split(num_rels, dim=0)
            union_spatial_splits = union_features[1].split(num_rels, dim=0)
        else:
            union_vis_splits = union_features.split(num_rels, dim=0) if union_features is not None else [None] * len(num_rels)
            union_spatial_splits = [None] * len(num_rels)

        rel_inputs = []
        pair_preds = []

        for pair_idx, cur_union_vis, cur_union_spatial, cur_obj_pred, cur_obj_embed in zip(
            rel_pair_idxs, union_vis_splits, union_spatial_splits, obj_preds, obj_embed_ctxs
        ):
            if pair_idx.numel() == 0:
                rel_inputs.append(cur_obj_embed.new_zeros((0, self.hidden_dim)))
                pair_preds.append(cur_obj_pred.new_zeros((0, 2)))
                continue

            pair_sem = torch.cat((cur_obj_embed[pair_idx[:, 0]], cur_obj_embed[pair_idx[:, 1]]), dim=-1)
            pair_sem = self.pair_semantic(pair_sem)
            pair_preds.append(torch.stack((cur_obj_pred[pair_idx[:, 0]], cur_obj_pred[pair_idx[:, 1]]), dim=1))

            union_vis = self.union_visual_proj(cur_union_vis)
            if cur_union_spatial is None:
                union_spatial = union_vis.new_zeros(union_vis.shape)
            else:
                union_spatial = self.union_spatial_proj(cur_union_spatial)

            rel_sum = union_vis + union_spatial + pair_sem
            rel_inputs.append(self.rel_input_proj(rel_sum))

        flat_rel_inputs = cat(rel_inputs, dim=0) if len(rel_inputs) > 0 else roi_features.new_zeros((0, self.hidden_dim))
        rel_ctx = self.rel_encoder(flat_rel_inputs, num_rels) if flat_rel_inputs.numel() > 0 else flat_rel_inputs
        rel_dists = self.rel_classifier(rel_ctx) if rel_ctx.numel() > 0 else rel_ctx.new_zeros((0, self.num_rel_cls))
        if self.use_bias and rel_ctx.numel() > 0:
            pair_pred = cat(pair_preds, dim=0)
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        return list(obj_dists), list(rel_dists.split(num_rels, dim=0)), {}
