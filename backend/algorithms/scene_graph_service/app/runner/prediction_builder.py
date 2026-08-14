from __future__ import annotations

from typing import Any

from services.scene_graph_service.app.postprocess.relation_postprocess import (
    serialize_objects,
    serialize_relationships,
)
from services.scene_graph_service.app.runner.base_runner import SceneGraphPredictionResult


def build_prediction_result(
    *,
    image_id: str,
    image_path: str,
    input_objects: list[dict[str, Any]],
    bboxes_xyxy: list[list[float]],
    pred_label_ids: list[int],
    pred_scores: list[float],
    rel_pair_idxs: list[list[int]],
    pred_rel_scores: list[list[float]],
    pred_rel_label_ids: list[int],
    idx_to_label: dict[int, str],
    idx_to_predicate: dict[int, str],
    meta: dict[str, Any] | None = None,
    min_rel_score: float = 0.0,
    topk_relations: int = 0,
) -> SceneGraphPredictionResult:
    objects = serialize_objects(
        bboxes_xyxy=bboxes_xyxy,
        pred_label_ids=pred_label_ids,
        pred_scores=pred_scores,
        input_objects=input_objects,
        idx_to_label=idx_to_label,
    )
    relationships = serialize_relationships(
        rel_pair_idxs=rel_pair_idxs,
        pred_rel_scores=pred_rel_scores,
        pred_rel_label_ids=pred_rel_label_ids,
        objects=objects,
        idx_to_predicate=idx_to_predicate,
        min_rel_score=min_rel_score,
        topk_relations=topk_relations,
    )
    return SceneGraphPredictionResult(
        image_id=image_id,
        image_path=image_path,
        objects=objects,
        relationships=relationships,
        meta=meta or {},
    )
