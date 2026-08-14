from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from services.scene_graph_service.app.data.vg_h5_dataset import VGH5DatasetLite, VGH5Sample
from services.scene_graph_service.app.eval.service_evaluator import (
    ServiceSGGEvaluationConfig,
    ServiceSGGEvaluator,
)


def collect_triplets(dataset: VGH5DatasetLite) -> set[tuple[int, int, int]]:
    triplets: set[tuple[int, int, int]] = set()
    for gt_classes, rels in zip(dataset.gt_classes, dataset.relationships):
        for sub_idx, obj_idx, pred_label in rels:
            triplets.add(
                (
                    int(gt_classes[int(sub_idx)]),
                    int(gt_classes[int(obj_idx)]),
                    int(pred_label),
                )
            )
    return triplets


def build_zeroshot_triplets(train_dataset: VGH5DatasetLite, eval_dataset: VGH5DatasetLite) -> np.ndarray:
    train_triplets = collect_triplets(train_dataset)
    eval_triplets = collect_triplets(eval_dataset)
    zeroshot_triplets = sorted(eval_triplets - train_triplets)
    if len(zeroshot_triplets) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(zeroshot_triplets, dtype=np.int64)


def load_prediction_payload(prediction_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(prediction_path).read_text(encoding="utf-8"))


def _build_rel_score_vector(rel: dict[str, Any], num_rel_category: int) -> list[float]:
    all_scores = rel.get("all_scores")
    if isinstance(all_scores, list) and len(all_scores) == num_rel_category:
        return [float(v) for v in all_scores]

    predicate_id = int(rel["predicate_id"])
    score = float(rel.get("score", 1.0))
    row = [0.0] * num_rel_category
    if 0 <= predicate_id < num_rel_category:
        row[predicate_id] = score
    return row


def build_local_container_from_payload(
    sample: VGH5Sample,
    prediction_payload: dict[str, Any],
    num_rel_category: int,
) -> dict[str, np.ndarray]:
    gt_rels = np.asarray(sample.relationships, dtype=np.int64)
    gt_boxes = np.asarray(sample.boxes_xyxy, dtype=np.float32)
    gt_classes = np.asarray(sample.labels, dtype=np.int64)

    objects = prediction_payload.get("objects", [])
    relationships = prediction_payload.get("relationships", [])
    pred_boxes = np.asarray([obj["bbox"] for obj in objects], dtype=np.float32) if objects else np.zeros((0, 4), dtype=np.float32)
    pred_classes = np.asarray([int(obj["label_id"]) for obj in objects], dtype=np.int64) if objects else np.zeros((0,), dtype=np.int64)
    obj_scores = np.asarray([float(obj.get("score", 1.0)) for obj in objects], dtype=np.float32) if objects else np.zeros((0,), dtype=np.float32)

    pred_rel_inds = (
        np.asarray([[int(rel["subject_index"]), int(rel["object_index"])] for rel in relationships], dtype=np.int64)
        if relationships
        else np.zeros((0, 2), dtype=np.int64)
    )
    rel_scores = (
        np.asarray([_build_rel_score_vector(rel, num_rel_category) for rel in relationships], dtype=np.float32)
        if relationships
        else np.zeros((0, num_rel_category), dtype=np.float32)
    )

    return {
        "gt_rels": gt_rels,
        "gt_boxes": gt_boxes,
        "gt_classes": gt_classes,
        "pred_rel_inds": pred_rel_inds,
        "rel_scores": rel_scores,
        "pred_boxes": pred_boxes,
        "pred_classes": pred_classes,
        "obj_scores": obj_scores,
    }


def evaluate_predictions_with_service_evaluator(
    *,
    eval_dataset: VGH5DatasetLite,
    prediction_dir: str | Path,
    train_dataset: VGH5DatasetLite | None = None,
    mode: str = "predcls",
    iou_thres: float = 0.5,
    eval_all_metrics: bool = True,
    print_detail: bool = True,
) -> tuple[str, dict[str, object]]:
    zeroshot_triplets = None if train_dataset is None else build_zeroshot_triplets(train_dataset, eval_dataset)
    config = ServiceSGGEvaluationConfig(
        mode=mode,
        iou_thres=iou_thres,
        num_rel_category=len(eval_dataset.ind_to_predicates),
        eval_all_metrics=eval_all_metrics,
        zeroshot_triplet=zeroshot_triplets,
        print_detail=print_detail,
    )
    evaluator = ServiceSGGEvaluator(config, eval_dataset.ind_to_predicates)
    global_container = evaluator.build_global_container()

    prediction_dir = Path(prediction_dir)
    for index in range(len(eval_dataset)):
        sample = eval_dataset[index]
        prediction_path = prediction_dir / f"{sample.image_id}.json"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found for image_id={sample.image_id}: {prediction_path}")

        payload = load_prediction_payload(prediction_path)
        local_container = build_local_container_from_payload(sample, payload, config.num_rel_category)
        if len(local_container["gt_rels"]) == 0 or local_container["pred_rel_inds"].shape[0] == 0:
            continue

        if mode != "sgdet":
            evaluator.evaluator["eval_pair_accuracy"].prepare_gtpair(local_container)
        if eval_all_metrics:
            evaluator.evaluator["eval_zeroshot_recall"].prepare_zeroshot(global_container, local_container)
            evaluator.evaluator["eval_ng_zeroshot_recall"].prepare_zeroshot(global_container, local_container)

        if mode == "predcls":
            local_container["pred_boxes"] = local_container["gt_boxes"]
            local_container["pred_classes"] = local_container["gt_classes"]
            local_container["obj_scores"] = np.ones(local_container["gt_classes"].shape[0], dtype=np.float32)

        local_container = evaluator.evaluator["eval_recall"].calculate_recall(global_container, local_container, mode)
        if eval_all_metrics:
            evaluator.evaluator["eval_nog_recall"].calculate_recall(global_container, local_container, mode)
            evaluator.evaluator["eval_pair_accuracy"].calculate_recall(global_container, local_container, mode)
        evaluator.evaluator["eval_mean_recall"].collect_mean_recall_items(global_container, local_container, mode)
        if eval_all_metrics:
            evaluator.evaluator["eval_ng_mean_recall"].collect_mean_recall_items(global_container, local_container, mode)
            evaluator.evaluator["eval_zeroshot_recall"].calculate_recall(global_container, local_container, mode)
            evaluator.evaluator["eval_ng_zeroshot_recall"].calculate_recall(global_container, local_container, mode)

    report = evaluator.generate_report()
    return report, evaluator.result_dict
