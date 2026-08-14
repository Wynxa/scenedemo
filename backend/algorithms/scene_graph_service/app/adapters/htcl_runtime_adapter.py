from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...bootstrap_runtime import bootstrap_runtime

from services.scene_graph_service.app.adapters.legacy_boxlist_bridge import lite_to_legacy_boxlist
from services.scene_graph_service.app.postprocess.relation_postprocess import (
    serialize_objects,
    serialize_relationships,
)
from services.scene_graph_service.app.services.request_preprocessor import prepare_inference_input

bootstrap_runtime()

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer


@dataclass
class LegacyHtclConfig:
    config_file: str
    weight_file: str
    label_dict: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    min_rel_score: float = 0.0
    topk_relations: int = 0
    opts: list[str] | None = None


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_label_dict(path: str | Path) -> tuple[dict[int, str], dict[int, str], dict[str, int]]:
    payload = _load_json(path)
    idx_to_label = {int(k): v for k, v in payload["idx_to_label"].items()}
    idx_to_predicate = {int(k): v for k, v in payload["idx_to_predicate"].items()}
    label_to_idx = payload["label_to_idx"]
    return idx_to_label, idx_to_predicate, label_to_idx


class HtclRuntimeAdapter:
    def __init__(self, runtime_config: LegacyHtclConfig) -> None:
        self.runtime_config = runtime_config
        self.idx_to_label, self.idx_to_predicate, self.label_to_idx = _load_label_dict(runtime_config.label_dict)
        self.model = self._load_model()

    def _load_model(self):
        cfg.merge_from_file(self.runtime_config.config_file)
        if self.runtime_config.opts:
            cfg.merge_from_list(self.runtime_config.opts)
        cfg.defrost()
        cfg.MODEL.WEIGHT = self.runtime_config.weight_file
        cfg.MODEL.DEVICE = self.runtime_config.device
        cfg.freeze()

        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        model.eval()
        DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHT)
        return model

    def infer_boxlist(self, image_path: str, objects: list[dict[str, Any]]):
        prepared = prepare_inference_input(
            image_path=image_path,
            objects=objects,
            label_to_idx=self.label_to_idx,
            min_size=cfg.INPUT.MIN_SIZE_TEST,
            max_size=cfg.INPUT.MAX_SIZE_TEST,
            pixel_mean=cfg.INPUT.PIXEL_MEAN,
            pixel_std=cfg.INPUT.PIXEL_STD,
            to_bgr255=cfg.INPUT.TO_BGR255,
        )
        resized_target = lite_to_legacy_boxlist(prepared.resized_target).to(cfg.MODEL.DEVICE)

        with torch.no_grad():
            outputs = self.model([prepared.image_tensor.to(cfg.MODEL.DEVICE)], [resized_target])

        result = outputs[0].to(torch.device("cpu"))
        if result.size != prepared.pil_image.size:
            result = result.resize(prepared.pil_image.size)
        return result

    def infer(self, image_path: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
        result = self.infer_boxlist(image_path, objects)

        bboxes_xyxy = [[float(v) for v in row] for row in result.convert("xyxy").bbox.tolist()]
        pred_label_ids = [int(v) for v in result.get_field("pred_labels").tolist()]
        pred_scores = [float(v) for v in result.get_field("pred_scores").tolist()]
        objects_payload = serialize_objects(
            bboxes_xyxy=bboxes_xyxy,
            pred_label_ids=pred_label_ids,
            pred_scores=pred_scores,
            input_objects=objects,
            idx_to_label=self.idx_to_label,
        )
        relationships_payload = serialize_relationships(
            rel_pair_idxs=result.get_field("rel_pair_idxs").tolist(),
            pred_rel_scores=result.get_field("pred_rel_scores").tolist(),
            pred_rel_label_ids=result.get_field("pred_rel_labels").tolist(),
            objects=objects_payload,
            idx_to_predicate=self.idx_to_predicate,
            min_rel_score=self.runtime_config.min_rel_score,
            topk_relations=self.runtime_config.topk_relations,
        )
        return {
            "objects": objects_payload,
            "relationships": relationships_payload,
        }
