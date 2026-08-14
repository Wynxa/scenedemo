from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from services.scene_graph_service.app.adapters.legacy_visual_backbone_adapter import (
    LegacyVisualBackboneAdapter,
    LegacyVisualBackboneConfig,
)
from services.scene_graph_service.app.config import ServiceOwnedRuntimeConfig
from services.scene_graph_service.app.data import VGH5DatasetLite, VGH5Sample
from services.scene_graph_service.app.modules.relation_head_lite import (
    RelationHeadLite,
    load_relation_head_from_checkpoint,
)
from services.scene_graph_service.app.modules.relation_predictors_lite import (
    PENETHTCLLite,
    build_lite_htcl_cfg,
)
from services.scene_graph_service.app.runner.base_runner import (
    SceneGraphPredictionBundle,
    SceneGraphPredictionResult,
)
from services.scene_graph_service.app.runner.checkpoint_inspector import inspect_checkpoint
from services.scene_graph_service.app.runner.prediction_builder import build_prediction_result
from services.scene_graph_service.app.services.request_preprocessor import (
    prepare_inference_input,
)


@dataclass
class HTCLServiceRunner:
    runtime_config: ServiceOwnedRuntimeConfig

    def __post_init__(self) -> None:
        ind_to_classes, ind_to_predicates, _ = VGH5DatasetLite._load_info(self.runtime_config.paths.dict_file)
        self.idx_to_label = {i: name for i, name in enumerate(ind_to_classes)}
        self.idx_to_predicate = {i: name for i, name in enumerate(ind_to_predicates)}
        self.label_to_idx = {name: i for i, name in enumerate(ind_to_classes)}
        self._runtime_state: dict[str, Any] | None = None

    def ensure_runtime(self) -> None:
        if self._runtime_state is not None:
            return
        self._runtime_state = self._build_runtime_state()

    def _build_runtime_state(self) -> dict[str, Any]:
        checkpoint_summary = None
        if self.runtime_config.paths.weight_file:
            try:
                checkpoint_summary = inspect_checkpoint(self.runtime_config.paths.weight_file, map_location="cpu")
            except Exception as exc:  # pragma: no cover - diagnostic path
                checkpoint_summary = {
                    "checkpoint_path": self.runtime_config.paths.weight_file,
                    "error": str(exc),
                }
        return {
            "status": "hybrid_visual_relation_ready",
            "device": self.runtime_config.device,
            "weight_file": self.runtime_config.paths.weight_file,
            "checkpoint_summary": checkpoint_summary,
            "notes": [
                "service-owned dataset reader and relation head are active",
                "legacy runtime is retained only for backbone/FPN feature extraction",
                "upstream detector should pass image + boxes + labels into this service",
            ],
        }

    def _ensure_components(self) -> dict[str, Any]:
        self.ensure_runtime()
        if self._runtime_state is None:
            raise RuntimeError("Runtime state is not initialized.")
        components = self._runtime_state.get("components")
        if components is not None:
            return components

        if not self.runtime_config.paths.config_file:
            raise ValueError("runtime_config.paths.config_file is required for legacy backbone/FPN extraction.")
        if not self.runtime_config.paths.glove_dir:
            raise ValueError("runtime_config.paths.glove_dir is required for service-owned relation predictor.")

        backbone_adapter = LegacyVisualBackboneAdapter(
            LegacyVisualBackboneConfig(
                config_file=self.runtime_config.paths.config_file,
                opts_config_file=self.runtime_config.paths.opts_config_file,
                weight_file=self.runtime_config.paths.weight_file,
                device=self.runtime_config.device,
                glove_dir=self.runtime_config.paths.glove_dir,
                img_dir=self.runtime_config.paths.img_dir,
                roidb_file=self.runtime_config.paths.roidb_file,
                dict_file=self.runtime_config.paths.dict_file,
                image_file=self.runtime_config.paths.image_file,
                conflict_groups_json=self.runtime_config.paths.conflict_groups_json,
            )
        )
        predictor_cfg = build_lite_htcl_cfg(
            glove_dir=self.runtime_config.paths.glove_dir,
            num_obj_classes=len(self.idx_to_label),
            num_rel_classes=len(self.idx_to_predicate),
            conflict_groups_json=self.runtime_config.paths.conflict_groups_json,
        )
        predictor = PENETHTCLLite(
            cfg=predictor_cfg,
            in_channels=4096,
            obj_classes=[self.idx_to_label[i] for i in range(len(self.idx_to_label))],
            rel_classes=[self.idx_to_predicate[i] for i in range(len(self.idx_to_predicate))],
        )
        relation_head = RelationHeadLite(predictor=predictor)
        checkpoint = torch.load(self.runtime_config.paths.weight_file, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        load_summary = load_relation_head_from_checkpoint(relation_head, state_dict)
        relation_head.to(backbone_adapter.device)
        relation_head.eval()

        components = {
            "backbone_adapter": backbone_adapter,
            "relation_head": relation_head,
            "relation_head_load_summary": load_summary,
        }
        self._runtime_state["components"] = components
        self._runtime_state["relation_head_load_summary"] = load_summary
        return components

    def describe_runtime(self) -> dict[str, Any]:
        self.ensure_runtime()
        return dict(self._runtime_state or {})

    def _sample_to_input_objects(self, sample: VGH5Sample) -> list[dict[str, Any]]:
        return [
            {
                "object_id": idx,
                "label_id": int(label_id),
                "bbox": [float(v) for v in bbox],
                "score": 1.0,
                "source": "service_dataset",
            }
            for idx, (bbox, label_id) in enumerate(zip(sample.boxes_xyxy, sample.labels))
        ]

    def prepare_sample(self, sample: VGH5Sample) -> dict[str, Any]:
        input_objects = self._sample_to_input_objects(sample)
        prepared = prepare_inference_input(
            image_path=sample.image_path,
            objects=input_objects,
            label_to_idx={k: int(v) for k, v in self.label_to_idx.items()},
            min_size=self.runtime_config.preprocess.min_size,
            max_size=self.runtime_config.preprocess.max_size,
            pixel_mean=self.runtime_config.preprocess.pixel_mean,
            pixel_std=self.runtime_config.preprocess.pixel_std,
            to_bgr255=self.runtime_config.preprocess.to_bgr255,
        )
        return {
            "sample": sample,
            "input_objects": input_objects,
            "prepared": prepared,
        }

    def prepare_request(
        self,
        *,
        image_path: str,
        objects: list[dict[str, Any]],
        image_id: str | None = None,
    ) -> dict[str, Any]:
        prepared = prepare_inference_input(
            image_path=image_path,
            objects=objects,
            label_to_idx={k: int(v) for k, v in self.label_to_idx.items()},
            min_size=self.runtime_config.preprocess.min_size,
            max_size=self.runtime_config.preprocess.max_size,
            pixel_mean=self.runtime_config.preprocess.pixel_mean,
            pixel_std=self.runtime_config.preprocess.pixel_std,
            to_bgr255=self.runtime_config.preprocess.to_bgr255,
        )
        sample = VGH5Sample(
            index=-1,
            image_id=image_id or image_path,
            image_path=image_path,
            width=prepared.pil_image.size[0],
            height=prepared.pil_image.size[1],
            boxes_xyxy=[list(map(float, obj["bbox"])) for obj in objects],
            labels=[int(obj.get("label_id", 0)) for obj in objects],
            attributes=[[0] for _ in objects],
            relationships=[],
        )
        return {
            "sample": sample,
            "input_objects": objects,
            "prepared": prepared,
        }

    def _build_test_pairs(self, num_objects: int, device: torch.device) -> torch.Tensor:
        cand_matrix = torch.ones((num_objects, num_objects), device=device) - torch.eye(num_objects, device=device)
        idxs = torch.nonzero(cand_matrix).view(-1, 2)
        if len(idxs) == 0:
            return torch.zeros((1, 2), dtype=torch.int64, device=device)
        return idxs.long()

    def _postprocess_relation_logits(
        self,
        *,
        obj_logits: torch.Tensor,
        rel_logits: torch.Tensor,
        rel_pair_idx: torch.Tensor,
    ) -> dict[str, Any]:
        obj_class_prob = torch.softmax(obj_logits, dim=-1)
        obj_class_prob[:, 0] = 0
        obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
        obj_pred = obj_pred + 1

        obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
        obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
        rel_class_prob = torch.softmax(rel_logits, dim=-1)
        rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
        rel_class = rel_class + 1
        triple_scores = rel_scores * obj_scores0 * obj_scores1
        _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
        return {
            "pred_label_ids": obj_pred.tolist(),
            "pred_scores": obj_scores.tolist(),
            "rel_pair_idxs": rel_pair_idx[sorting_idx].tolist(),
            "pred_rel_scores": rel_class_prob[sorting_idx].tolist(),
            "pred_rel_label_ids": rel_class[sorting_idx].tolist(),
        }

    def _predict_prepared(self, prepared_sample: dict[str, Any]) -> SceneGraphPredictionResult:
        sample: VGH5Sample = prepared_sample["sample"]
        input_objects: list[dict[str, Any]] = prepared_sample["input_objects"]
        prepared = prepared_sample["prepared"]
        components = self._ensure_components()
        backbone_adapter: LegacyVisualBackboneAdapter = components["backbone_adapter"]
        relation_head: RelationHeadLite = components["relation_head"]

        resized_target = prepared.resized_target.to(backbone_adapter.device)
        rel_pair_idx = self._build_test_pairs(len(resized_target), backbone_adapter.device)
        with torch.no_grad():
            features = backbone_adapter.extract_backbone_features(prepared.image_tensor)
            head_output = relation_head(
                features=features,
                proposals=[resized_target],
                rel_pair_idxs=[rel_pair_idx],
                rel_labels=None,
            )
        obj_logits = head_output.obj_logits[0].detach().cpu()
        rel_logits = head_output.rel_logits[0].detach().cpu()
        rel_pair_idx_cpu = rel_pair_idx.detach().cpu()
        postprocessed = self._postprocess_relation_logits(
            obj_logits=obj_logits,
            rel_logits=rel_logits,
            rel_pair_idx=rel_pair_idx_cpu,
        )
        resized_boxes = resized_target.to(torch.device("cpu")).bbox.tolist()
        scale_x = sample.width / prepared.resized_target.size[0]
        scale_y = sample.height / prepared.resized_target.size[1]
        bboxes_xyxy = [
            [
                float(box[0] * scale_x),
                float(box[1] * scale_y),
                float(box[2] * scale_x),
                float(box[3] * scale_y),
            ]
            for box in resized_boxes
        ]
        return self.build_result_from_raw_outputs(
            sample=sample,
            input_objects=input_objects,
            bboxes_xyxy=bboxes_xyxy,
            pred_label_ids=postprocessed["pred_label_ids"],
            pred_scores=postprocessed["pred_scores"],
            rel_pair_idxs=postprocessed["rel_pair_idxs"],
            pred_rel_scores=postprocessed["pred_rel_scores"],
            pred_rel_label_ids=postprocessed["pred_rel_label_ids"],
            meta={
                "runner": "htcl_service_hybrid",
                "visual_runtime": "legacy_backbone_fpn",
                "relation_runtime": "service_owned_relation_head",
            },
        )

    def predict_dataset(self, *, split: str, num_im: int = -1) -> SceneGraphPredictionBundle:
        self.ensure_runtime()
        dataset = VGH5DatasetLite(
            img_dir=self.runtime_config.paths.img_dir,
            roidb_file=self.runtime_config.paths.roidb_file,
            dict_file=self.runtime_config.paths.dict_file,
            image_file=self.runtime_config.paths.image_file,
            split=split,
            num_im=num_im,
        )
        predictions: list[SceneGraphPredictionResult] = []
        for sample in dataset:
            prepared_sample = self.prepare_sample(sample)
            predictions.append(self._predict_prepared(prepared_sample))
        return SceneGraphPredictionBundle(
            predictions=predictions,
            meta={
                "runner": "htcl_service",
                "split": split,
                "num_images": len(predictions),
            },
        )

    def predict_request(
        self,
        *,
        image_path: str,
        objects: list[dict[str, Any]],
        image_id: str | None = None,
    ) -> SceneGraphPredictionResult:
        prepared_sample = self.prepare_request(image_path=image_path, objects=objects, image_id=image_id)
        return self._predict_prepared(prepared_sample)

    def build_result_from_raw_outputs(
        self,
        *,
        sample: VGH5Sample,
        input_objects: list[dict[str, Any]],
        bboxes_xyxy: list[list[float]],
        pred_label_ids: list[int],
        pred_scores: list[float],
        rel_pair_idxs: list[list[int]],
        pred_rel_scores: list[list[float]],
        pred_rel_label_ids: list[int],
        meta: dict[str, Any] | None = None,
    ) -> SceneGraphPredictionResult:
        return build_prediction_result(
            image_id=sample.image_id,
            image_path=sample.image_path,
            input_objects=input_objects,
            bboxes_xyxy=bboxes_xyxy,
            pred_label_ids=pred_label_ids,
            pred_scores=pred_scores,
            rel_pair_idxs=rel_pair_idxs,
            pred_rel_scores=pred_rel_scores,
            pred_rel_label_ids=pred_rel_label_ids,
            idx_to_label=self.idx_to_label,
            idx_to_predicate=self.idx_to_predicate,
            meta=meta,
        )

