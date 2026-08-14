from __future__ import annotations

from typing import Any

import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite


def lite_to_legacy_boxlist(boxlist: BoxListLite) -> BoxList:
    legacy = BoxList(boxlist.bbox, boxlist.size, mode=boxlist.mode)
    for key, value in boxlist.extra_fields.items():
        legacy.add_field(key, value)
    return legacy


def legacy_to_lite_boxlist(boxlist: BoxList) -> BoxListLite:
    lite = BoxListLite(boxlist.bbox, boxlist.size, mode=boxlist.mode)
    fields = getattr(boxlist, "extra_fields", {})
    triplet_fields = getattr(boxlist, "triplet_extra_fields", [])
    for key, value in fields.items():
        is_triplet = key in triplet_fields
        if isinstance(value, torch.Tensor):
            lite.add_field(key, value, is_triplet=is_triplet)
        else:
            lite.add_field(key, value, is_triplet=is_triplet)
    return lite
