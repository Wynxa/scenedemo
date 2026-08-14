from collections import OrderedDict
from typing import Any


def xyxy_to_xywh(bbox: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return {
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1,
    }


def serialize_objects(
    bboxes_xyxy: list[list[float]],
    pred_label_ids: list[int],
    pred_scores: list[float],
    input_objects: list[dict[str, Any]],
    idx_to_label: dict[int, str],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for idx, bbox_xyxy in enumerate(bboxes_xyxy):
        label_id = int(pred_label_ids[idx])
        input_item = input_objects[idx] if idx < len(input_objects) else {}
        bbox_xywh = xyxy_to_xywh(bbox_xyxy)

        row = OrderedDict()
        row["object_index"] = idx
        row["object_id"] = input_item.get("object_id", idx)
        row["name"] = idx_to_label[label_id]
        row["label_id"] = label_id
        row["x"] = bbox_xywh["x"]
        row["y"] = bbox_xywh["y"]
        row["w"] = bbox_xywh["w"]
        row["h"] = bbox_xywh["h"]
        row["bbox"] = [float(v) for v in bbox_xyxy]
        row["bbox_xyxy"] = [float(v) for v in bbox_xyxy]
        row["score"] = float(pred_scores[idx])
        if "score" in input_item:
            row["det_score"] = float(input_item["score"])
        elif "det_score" in input_item:
            row["det_score"] = float(input_item["det_score"])
        if "source" in input_item:
            row["source"] = input_item["source"]
        objects.append(row)
    return objects


def serialize_relationships(
    rel_pair_idxs: list[list[int]],
    pred_rel_scores: list[list[float]],
    pred_rel_label_ids: list[int],
    objects: list[dict[str, Any]],
    idx_to_predicate: dict[int, str],
    min_rel_score: float = 0.0,
    topk_relations: int = 0,
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    for rel_idx, pair in enumerate(rel_pair_idxs):
        predicate_id = int(pred_rel_label_ids[rel_idx])
        predicate_score = float(pred_rel_scores[rel_idx][predicate_id])
        if predicate_score < min_rel_score:
            continue

        subject_index = int(pair[0])
        object_index = int(pair[1])
        row = OrderedDict()
        row["relationship_id"] = len(relationships)
        row["subject_index"] = subject_index
        row["object_index"] = object_index
        row["subject_id"] = objects[subject_index]["object_id"]
        row["object_id"] = objects[object_index]["object_id"]
        row["subject_name"] = objects[subject_index]["name"]
        row["object_name"] = objects[object_index]["name"]
        row["predicate_id"] = predicate_id
        row["predicate"] = idx_to_predicate[predicate_id]
        row["score"] = predicate_score
        row["all_scores"] = [float(v) for v in pred_rel_scores[rel_idx]]
        relationships.append(row)

    relationships.sort(key=lambda item: item["score"], reverse=True)
    if topk_relations > 0:
        relationships = relationships[:topk_relations]
        for idx, row in enumerate(relationships):
            row["relationship_id"] = idx
    return relationships
