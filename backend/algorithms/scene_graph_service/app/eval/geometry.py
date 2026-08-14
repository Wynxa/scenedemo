from __future__ import annotations

from functools import reduce

import numpy as np


def intersect_2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same number of columns.")
    return (x1[..., None] == x2.T[None, ...]).all(1)


def argsort_desc(scores: np.ndarray) -> np.ndarray:
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def bbox_overlaps(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt + 1.0, a_min=0.0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1.0) * (boxes1[:, 3] - boxes1[:, 1] + 1.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1.0) * (boxes2[:, 3] - boxes2[:, 1] + 1.0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def triplet(
    relations: np.ndarray,
    classes: np.ndarray,
    boxes: np.ndarray,
    predicate_scores: np.ndarray | None = None,
    class_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    sub_id, obj_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[obj_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[obj_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((class_scores[sub_id], predicate_scores, class_scores[obj_id]))
    return triplets, triplet_boxes, triplet_scores


def compute_pred_matches(
    gt_triplets: np.ndarray,
    pred_triplets: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_thres: float,
    phrdet: bool = False,
) -> list[list[int]]:
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for _ in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(
        np.where(gt_has_match)[0],
        gt_boxes[gt_has_match],
        keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)
            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)
            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres
        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for pred_index in np.where(keep_inds)[0][inds]:
            pred_to_gt[pred_index].append(int(gt_ind))
    return pred_to_gt


def rel_nms(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_rel_inds: np.ndarray,
    rel_scores: np.ndarray,
    nms_thresh: float = 0.5,
    l21_thr: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    ious = bbox_overlaps(pred_boxes, pred_boxes)
    sub_ious = ious[pred_rel_inds[:, 0]][:, pred_rel_inds[:, 0]]
    obj_ious = ious[pred_rel_inds[:, 1]][:, pred_rel_inds[:, 1]]
    rel_ious = np.minimum(sub_ious, obj_ious)
    sub_labels = pred_classes[pred_rel_inds[:, 0]]
    obj_labels = pred_classes[pred_rel_inds[:, 1]]

    l21 = np.sqrt((np.power(rel_scores[:, None, :], 2.0) + np.power(rel_scores[None, :, :], 2.0))).sum(axis=-1)
    is_overlap = (
        (rel_ious >= nms_thresh)
        & (sub_labels[:, None] == sub_labels[None, :])
        & (obj_labels[:, None] == obj_labels[None, :])
        & (l21 > l21_thr)
    )
    is_overlap = is_overlap[:, :, None].repeat(rel_scores.shape[1], axis=2)

    rel_scores_cp = rel_scores.copy()
    rel_scores_cp[:, 0] = 0.0
    pred_rels = np.zeros(rel_scores_cp.shape[0], dtype=np.int64)

    for _ in range(rel_scores_cp.shape[0]):
        box_ind, cls_ind = np.unravel_index(rel_scores_cp.argmax(), rel_scores_cp.shape)
        if float(pred_rels[int(box_ind)]) == 0:
            pred_rels[int(box_ind)] = int(cls_ind)
        rel_scores_cp[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
        rel_scores_cp[box_ind] = -1.0

    rel_pick_scores = rel_scores[np.arange(pred_rels.shape[0], dtype=np.int64), pred_rels]
    return pred_rels, rel_pick_scores


def reduce_union(pred_to_gt: list[list[int]], k: int) -> np.ndarray:
    selected = pred_to_gt[:k]
    if len(selected) == 0:
        return np.array([], dtype=np.int64)
    return reduce(np.union1d, selected)
