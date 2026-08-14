from __future__ import annotations

from typing import List


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def classwise_nms(objects: List[dict], iou_threshold: float) -> List[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in objects:
        grouped.setdefault(item["label"], []).append(item)

    kept: List[dict] = []
    for _, rows in grouped.items():
        rows = sorted(rows, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        class_kept: List[dict] = []
        while rows:
            current = rows.pop(0)
            class_kept.append(current)
            rows = [
                row for row in rows
                if compute_iou(current["bbox"], row["bbox"]) < iou_threshold
            ]
        kept.extend(class_kept)
    return kept
