from __future__ import annotations

import torch

from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite


def boxlist_iou_lite(boxlist1: BoxListLite, boxlist2: BoxListLite) -> torch.Tensor:
    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1.size, boxlist2.size))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    to_remove = 1
    wh = (rb - lt + to_remove).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)


def boxlist_union_lite(boxlist1: BoxListLite, boxlist2: BoxListLite) -> BoxListLite:
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    union_box = torch.cat(
        (
            torch.min(boxlist1.bbox[:, :2], boxlist2.bbox[:, :2]),
            torch.max(boxlist1.bbox[:, 2:], boxlist2.bbox[:, 2:]),
        ),
        dim=1,
    )
    return BoxListLite(union_box, boxlist1.size, "xyxy")


def boxlist_intersection_lite(boxlist1: BoxListLite, boxlist2: BoxListLite) -> BoxListLite:
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    inter_box = torch.cat(
        (
            torch.max(boxlist1.bbox[:, :2], boxlist2.bbox[:, :2]),
            torch.min(boxlist1.bbox[:, 2:], boxlist2.bbox[:, 2:]),
        ),
        dim=1,
    )
    invalid = torch.max((inter_box[:, 0] >= inter_box[:, 2]).long(), (inter_box[:, 1] >= inter_box[:, 3]).long())
    inter_box[invalid > 0] = 0
    return BoxListLite(inter_box, boxlist1.size, "xyxy")
