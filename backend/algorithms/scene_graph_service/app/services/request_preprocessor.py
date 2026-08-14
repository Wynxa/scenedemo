from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image
import torch

from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite
from services.scene_graph_service.app.transforms.image_transforms import build_inference_transforms


def normalize_bbox(item: dict[str, Any]) -> list[float]:
    bbox = item.get("bbox") or item.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        raise ValueError("Each object must contain bbox or bbox_xyxy in xyxy format.")
    return [float(v) for v in bbox]


def resolve_label_id(item: dict[str, Any], label_to_idx: dict[str, int]) -> int:
    if item.get("label_id") is not None:
        return int(item["label_id"])
    label_name = item.get("label") or item.get("name") or item.get("class_name") or item.get("category_name")
    if label_name is None:
        raise ValueError("Each object must contain label or label_id.")
    if label_name not in label_to_idx:
        raise ValueError("Unknown object label '{}'.".format(label_name))
    return int(label_to_idx[label_name])


@dataclass
class PreparedInferenceInput:
    pil_image: Image.Image
    image_tensor: torch.Tensor
    target: BoxListLite
    resized_target: BoxListLite


def build_target_boxlist(
    objects: list[dict[str, Any]],
    image_size: tuple[int, int],
    label_to_idx: dict[str, int],
) -> BoxListLite:
    boxes = []
    labels = []
    for item in objects:
        boxes.append(normalize_bbox(item))
        labels.append(resolve_label_id(item, label_to_idx))

    target = BoxListLite(torch.tensor(boxes, dtype=torch.float32), image_size, mode="xyxy")
    target.add_field("labels", torch.tensor(labels, dtype=torch.int64))
    return target


def prepare_inference_input(
    *,
    image_path: str,
    objects: list[dict[str, Any]],
    label_to_idx: dict[str, int],
    min_size: int | tuple[int, ...],
    max_size: int,
    pixel_mean: list[float],
    pixel_std: list[float],
    to_bgr255: bool,
) -> PreparedInferenceInput:
    pil_image = Image.open(image_path).convert("RGB")
    target = build_target_boxlist(objects, pil_image.size, label_to_idx)
    transform = build_inference_transforms(
        min_size=min_size,
        max_size=max_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        to_bgr255=to_bgr255,
    )
    image_tensor, resized_target = transform(pil_image, target)
    if resized_target is None:
        raise RuntimeError("Expected resized target to be generated for precls inference.")
    return PreparedInferenceInput(
        pil_image=pil_image,
        image_tensor=image_tensor,
        target=target,
        resized_target=resized_target,
    )
