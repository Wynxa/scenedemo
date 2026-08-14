from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from services.hazard_reasoning_service.app.config import HazardReasoningRuntimeConfig
from services.hazard_reasoning_service.app.modules.context_config import load_context_config
from services.hazard_reasoning_service.app.modules.dataset import PrototypeInventory
from services.hazard_reasoning_service.app.modules.labeling import (
    build_worker_relations,
    matches_context_policy,
    resolve_proximity_conflicts,
)
from services.hazard_reasoning_service.app.modules.modeling import (
    DualEncoderModel,
    move_token_batch_to_device,
    predict_major_labels,
)
from services.hazard_reasoning_service.app.modules.scene_serializer import (
    build_scene_text_compact,
    build_scene_text_structured,
    build_worker_output_record,
)

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


@dataclass
class HazardInferenceArtifacts:
    tokenizer: object
    model: DualEncoderModel
    prototype_inventory: PrototypeInventory
    prototype_tokens: Dict[str, torch.Tensor]
    prototype_major_label_ids: torch.Tensor
    label_names: List[str]


class HazardReasoningService:
    WORK_CONTEXT_ZONE_NAMES = {
        "rebar_zone",
        "bricklaying_zone",
    }

    def __init__(self, runtime_config: HazardReasoningRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self.device = torch.device(runtime_config.device)
        self._artifacts: HazardInferenceArtifacts | None = None

    def _ensure_hf_env(self) -> None:
        cache_dir = self.runtime_config.paths.model_cache_dir
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

    def _build_tokenizer(self):
        if AutoTokenizer is None:
            raise ImportError("transformers is required for hazard reasoning inference.")
        return AutoTokenizer.from_pretrained(
            self.runtime_config.model.pretrained_model,
            cache_dir=self.runtime_config.paths.model_cache_dir or None,
            local_files_only=self.runtime_config.model.local_files_only,
        )

    def _tokenize_texts(self, tokenizer, texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.runtime_config.model.max_length,
            return_tensors="pt",
        )

    def _build_label_names(self, prototype_inventory: PrototypeInventory) -> List[str]:
        num_major_labels = len(prototype_inventory.major_label_to_id)
        label_names = ["unknown"] * num_major_labels
        for label_name, label_id in prototype_inventory.major_label_to_id.items():
            label_names[int(label_id)] = label_name
        return label_names

    def _load_artifacts(self) -> HazardInferenceArtifacts:
        self._ensure_hf_env()
        prototype_inventory = PrototypeInventory(self.runtime_config.paths.prototype_inventory)
        label_names = self._build_label_names(prototype_inventory)
        tokenizer = self._build_tokenizer()
        model = DualEncoderModel(
            pretrained_model_name=self.runtime_config.model.pretrained_model,
            projection_hidden_dim=self.runtime_config.model.projection_hidden_dim,
            projection_dim=self.runtime_config.model.projection_dim,
            dropout=self.runtime_config.model.dropout,
            share_backbone=self.runtime_config.model.share_backbone,
            cache_dir=self.runtime_config.paths.model_cache_dir or None,
            local_files_only=self.runtime_config.model.local_files_only,
        ).to(self.device)
        state_dict = torch.load(
            self.runtime_config.paths.checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        prototype_tokens = self._tokenize_texts(tokenizer, prototype_inventory.prototype_texts)
        prototype_tokens = move_token_batch_to_device(prototype_tokens, self.device)
        prototype_major_label_ids = prototype_inventory.prototype_major_label_ids.to(self.device)
        return HazardInferenceArtifacts(
            tokenizer=tokenizer,
            model=model,
            prototype_inventory=prototype_inventory,
            prototype_tokens=prototype_tokens,
            prototype_major_label_ids=prototype_major_label_ids,
            label_names=label_names,
        )

    def _ensure_artifacts(self) -> HazardInferenceArtifacts:
        if self._artifacts is None:
            self._artifacts = self._load_artifacts()
        return self._artifacts

    @staticmethod
    def _build_bbox_xyxy(obj: Dict) -> List[float]:
        if "bbox_xyxy" in obj:
            return [float(v) for v in obj["bbox_xyxy"]]
        x1 = float(obj["x"])
        y1 = float(obj["y"])
        x2 = float(obj["x"] + obj["w"])
        y2 = float(obj["y"] + obj["h"])
        return [x1, y1, x2, y2]

    def _select_context_policy(self, relations: List[Dict]) -> Dict:
        context_config = load_context_config(self.runtime_config.paths.context_policy)
        relation_pairs = self._build_context_relation_pairs(relations)
        contexts = sorted(context_config["contexts"], key=lambda item: int(item["priority"]), reverse=True)
        for item in contexts:
            triggers = [
                (rel["predicate"], rel["object_name"])
                for rel in item.get("trigger_relations", [])
            ]
            if matches_context_policy(relation_pairs, item["match_mode"], triggers):
                return item
        raise RuntimeError("No context policy matched worker sample.")

    def _build_context_relation_pairs(self, relations: List[Dict]) -> List[tuple[str, str]]:
        relation_pairs = [(rel["predicate"], rel["object_name"]) for rel in relations]
        expanded_pairs = list(relation_pairs)

        # In service-facing annotations, work-zone proximity is often marked as
        # `next_to` instead of `near`. For context selection we treat them as
        # equivalent so rebar/bricklaying context does not fall back to general work.
        for predicate, object_name in relation_pairs:
            if predicate == "next_to" and object_name in self.WORK_CONTEXT_ZONE_NAMES:
                expanded_pairs.append(("near", object_name))

        return expanded_pairs

    def _normalize_relations_for_reasoning_text(self, relations: List[Dict]) -> List[Dict]:
        normalized_relations: List[Dict] = []
        for rel in relations:
            normalized = dict(rel)
            if (
                normalized["predicate"] == "next_to"
                and normalized["object_name"] in self.WORK_CONTEXT_ZONE_NAMES
            ):
                normalized["predicate"] = "near"
            normalized_relations.append(normalized)
        return normalized_relations

    def build_worker_reasoning_samples(self, scene_graph_item: Dict) -> List[Dict]:
        objects = scene_graph_item.get("objects", [])
        relationships = scene_graph_item.get("relationships", [])
        worker_relations = build_worker_relations(objects, relationships)
        samples: List[Dict] = []

        for obj in objects:
            if obj.get("name") != "worker":
                continue

            worker_id = int(obj["object_id"])
            relations = resolve_proximity_conflicts(worker_relations.get(worker_id, []))
            reasoning_relations = self._normalize_relations_for_reasoning_text(relations)
            selected_policy = self._select_context_policy(relations)
            present_ppe = sorted(rel["object_name"] for rel in relations if rel["predicate"] == "wearing")
            required_ppe = list(selected_policy.get("required_ppe", []))
            missing_ppe = [ppe for ppe in required_ppe if ppe not in present_ppe]
            core_relations = [f"{rel['predicate']} {rel['object_name']}" for rel in reasoning_relations]

            sample = {
                "sample_id": f"{scene_graph_item['image_id']}_worker_{worker_id}",
                "image_id": scene_graph_item["image_id"],
                "file_name": scene_graph_item.get("file_name", ""),
                "worker_id": worker_id,
                "worker_bbox_xyxy": self._build_bbox_xyxy(obj),
                "context_id": selected_policy["context_id"],
                "required_ppe": required_ppe,
                "present_ppe": present_ppe,
                "missing_ppe": missing_ppe,
                "core_relations": core_relations,
                "relations": reasoning_relations,
                "raw_relations": relations,
            }
            sample["scene_text"] = build_scene_text_compact(sample)
            sample["scene_text_structured"] = build_scene_text_structured(
                sample,
                self.runtime_config.paths.context_policy,
            )
            samples.append(sample)
        return samples

    @staticmethod
    def compute_image_level_result(worker_results: List[Dict]) -> Dict:
        unsafe_workers = [row for row in worker_results if row["pred_major_label"] != "safe"]
        return {
            "has_unsafe_behavior": len(unsafe_workers) > 0,
            "unsafe_worker_count": len(unsafe_workers),
            "highest_risk_prediction": max(worker_results, key=lambda row: row["pred_score"], default=None),
            "unsafe_behavior_categories": HazardReasoningService.build_unsafe_behavior_categories(unsafe_workers),
        }

    @staticmethod
    def build_unsafe_behavior_categories(unsafe_workers: List[Dict]) -> List[Dict]:
        grouped: Dict[str, Dict] = {}
        for row in unsafe_workers:
            category = row["pred_major_label"]
            bucket = grouped.setdefault(
                category,
                {
                    "unsafe_behavior_category": category,
                    "worker_ids": [],
                    "worker_count": 0,
                },
            )
            bucket["worker_ids"].append(int(row["worker_id"]))
            bucket["worker_count"] += 1
        return list(grouped.values())

    def infer_from_scene_graph(self, scene_graph_item: Dict) -> Dict:
        samples = self.build_worker_reasoning_samples(scene_graph_item)
        base_meta = {
            "task": "hazard_reasoning_inference_from_scene_graph",
            "image_id": scene_graph_item["image_id"],
            "checkpoint": self.runtime_config.paths.checkpoint,
            "prototype_inventory": self.runtime_config.paths.prototype_inventory,
            "scene_text_field": self.runtime_config.model.scene_text_field,
            "pretrained_model": self.runtime_config.model.pretrained_model,
            "runtime": "hazard_reasoning_service",
        }

        if not samples:
            unsafe_behavior_results: List[Dict] = []
            return {
                "meta": base_meta,
                "scene_graph": scene_graph_item,
                "worker_results": [],
                "unsafe_behavior_results": unsafe_behavior_results,
                "image_level_result": {
                    "has_unsafe_behavior": False,
                    "unsafe_worker_count": 0,
                    "highest_risk_prediction": None,
                    "unsafe_behavior_categories": unsafe_behavior_results,
                },
            }

        artifacts = self._ensure_artifacts()
        scene_texts = [sample[self.runtime_config.model.scene_text_field] for sample in samples]
        scene_tokens = self._tokenize_texts(artifacts.tokenizer, scene_texts)
        scene_tokens = move_token_batch_to_device(scene_tokens, self.device)
        outputs = predict_major_labels(
            model=artifacts.model,
            scene_tokens=scene_tokens,
            prototype_tokens=artifacts.prototype_tokens,
            prototype_major_label_ids=artifacts.prototype_major_label_ids,
            num_major_labels=len(artifacts.prototype_inventory.major_label_to_id),
        )

        major_logits = outputs["major_logits"].detach().cpu()
        pred_ids = outputs["pred_major_label_ids"].detach().cpu().tolist()
        topk = min(self.runtime_config.model.topk, major_logits.size(1))
        topk_scores, topk_ids = torch.topk(torch.softmax(major_logits, dim=1), k=topk, dim=1)

        worker_results = []
        for idx, sample in enumerate(samples):
            pred_id = int(pred_ids[idx])
            topk_rows = [
                {
                    "label_id": int(label_id.item()),
                    "label": artifacts.label_names[int(label_id.item())],
                    "score": float(score.item()),
                }
                for score, label_id in zip(topk_scores[idx], topk_ids[idx])
            ]
            worker_results.append(
                build_worker_output_record(
                    sample=sample,
                    pred_label=artifacts.label_names[pred_id],
                    pred_label_id=pred_id,
                    pred_score=float(topk_scores[idx][0].item()),
                    topk=topk_rows,
                    context_policy_path=self.runtime_config.paths.context_policy,
                )
            )

        unsafe_behavior_results = self.build_unsafe_behavior_categories(
            [row for row in worker_results if row["pred_major_label"] != "safe"]
        )

        return {
            "meta": base_meta,
            "scene_graph": scene_graph_item,
            "worker_results": worker_results,
            "unsafe_behavior_results": unsafe_behavior_results,
            "image_level_result": {
                **self.compute_image_level_result(worker_results),
                "unsafe_behavior_categories": unsafe_behavior_results,
            },
        }
