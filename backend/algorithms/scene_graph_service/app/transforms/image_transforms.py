from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F

from services.scene_graph_service.app.structures.boxlist_lite import (
    BoxListLite,
    FLIP_LEFT_RIGHT,
    FLIP_TOP_BOTTOM,
)


@dataclass
class ResizeTransform:
    min_size: int | tuple[int, ...]
    max_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.min_size, (list, tuple)):
            self.min_size = (self.min_size,)

    def get_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        width, height = image_size
        size = self.min_size[0]
        min_original = float(min(width, height))
        max_original = float(max(width, height))
        if max_original / min_original * size > self.max_size:
            size = int(round(self.max_size * min_original / max_original))
        if (width <= height and width == size) or (height <= width and height == size):
            return height, width
        if width < height:
            out_width = size
            out_height = int(size * height / width)
        else:
            out_height = size
            out_width = int(size * width / height)
        return out_height, out_width

    def __call__(self, image: Image.Image, target: BoxListLite | None = None) -> tuple[Image.Image, BoxListLite | None]:
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image, None
        return image, target.resize(image.size)


@dataclass
class ToTensorTransform:
    def __call__(self, image: Image.Image, target: BoxListLite | None = None) -> tuple[torch.Tensor, BoxListLite | None]:
        return F.to_tensor(image), target


@dataclass
class NormalizeTransform:
    mean: list[float]
    std: list[float]
    to_bgr255: bool = True

    def __call__(self, image: torch.Tensor, target: BoxListLite | None = None) -> tuple[torch.Tensor, BoxListLite | None]:
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


@dataclass
class RandomHorizontalFlipTransform:
    prob: float = 0.0

    def __call__(self, image: Image.Image, target: BoxListLite | None = None) -> tuple[Image.Image, BoxListLite | None]:
        if self.prob <= 0:
            return image, target
        if np.random.rand() < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(FLIP_LEFT_RIGHT)
        return image, target


@dataclass
class RandomVerticalFlipTransform:
    prob: float = 0.0

    def __call__(self, image: Image.Image, target: BoxListLite | None = None) -> tuple[Image.Image, BoxListLite | None]:
        if self.prob <= 0:
            return image, target
        if np.random.rand() < self.prob:
            image = F.vflip(image)
            if target is not None:
                target = target.transpose(FLIP_TOP_BOTTOM)
        return image, target


class ComposeTransforms:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, image: Any, target: BoxListLite | None = None) -> tuple[Any, BoxListLite | None]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def build_inference_transforms(
    *,
    min_size: int = 600,
    max_size: int = 1000,
    pixel_mean: list[float] | None = None,
    pixel_std: list[float] | None = None,
    to_bgr255: bool = True,
) -> ComposeTransforms:
    return ComposeTransforms(
        [
            ResizeTransform(min_size=min_size, max_size=max_size),
            ToTensorTransform(),
            NormalizeTransform(
                mean=pixel_mean or [102.9801, 115.9465, 122.7717],
                std=pixel_std or [1.0, 1.0, 1.0],
                to_bgr255=to_bgr255,
            ),
        ]
    )
