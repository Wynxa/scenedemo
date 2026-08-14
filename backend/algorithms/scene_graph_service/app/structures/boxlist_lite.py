from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


@dataclass
class BoxListLite:
    bbox: torch.Tensor
    size: tuple[int, int]
    mode: str = "xyxy"
    extra_fields: dict[str, Any] = field(default_factory=dict)
    triplet_extra_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, torch.Tensor):
            self.bbox = torch.as_tensor(self.bbox, dtype=torch.float32)
        else:
            self.bbox = self.bbox.to(dtype=torch.float32)
        if self.bbox.ndimension() != 2:
            raise ValueError("bbox should have 2 dimensions, got {}".format(self.bbox.ndimension()))
        if self.bbox.size(-1) != 4:
            raise ValueError("last dimension of bbox should have size 4, got {}".format(self.bbox.size(-1)))
        if self.mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

    def add_field(self, name: str, value: Any, is_triplet: bool = False) -> None:
        self.extra_fields[name] = value
        if is_triplet and name not in self.triplet_extra_fields:
            self.triplet_extra_fields.append(name)

    def get_field(self, name: str) -> Any:
        return self.extra_fields[name]

    def has_field(self, name: str) -> bool:
        return name in self.extra_fields

    def fields(self) -> list[str]:
        return list(self.extra_fields.keys())

    def _split_into_xyxy(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "xyxy":
            return self.bbox.split(1, dim=-1)
        to_remove = 1
        xmin, ymin, w, h = self.bbox.split(1, dim=-1)
        return (
            xmin,
            ymin,
            xmin + (w - to_remove).clamp(min=0),
            ymin + (h - to_remove).clamp(min=0),
        )

    def _copy_extra_fields(self, source: "BoxListLite") -> None:
        for key, value in source.extra_fields.items():
            if key in source.triplet_extra_fields:
                self.add_field(key, value, is_triplet=True)
            else:
                self.add_field(key, value)

    def convert(self, mode: str) -> "BoxListLite":
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
        else:
            to_remove = 1
            bbox = torch.cat((xmin, ymin, xmax - xmin + to_remove, ymax - ymin + to_remove), dim=-1)
        converted = BoxListLite(bbox, self.size, mode=mode)
        converted._copy_extra_fields(self)
        return converted

    def resize(self, size: tuple[int, int]) -> "BoxListLite":
        ratios = tuple(float(s) / float(s0) for s, s0 in zip(size, self.size))
        if ratios[0] == ratios[1]:
            scaled_box = self.bbox * ratios[0]
            resized = BoxListLite(scaled_box, size, mode=self.mode)
        else:
            ratio_width, ratio_height = ratios
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
            scaled_box = torch.cat(
                (
                    xmin * ratio_width,
                    ymin * ratio_height,
                    xmax * ratio_width,
                    ymax * ratio_height,
                ),
                dim=-1,
            )
            resized = BoxListLite(scaled_box, size, mode="xyxy").convert(self.mode)
        for key, value in self.extra_fields.items():
            if hasattr(value, "resize") and not isinstance(value, torch.Tensor):
                value = value.resize(size)
            if key in self.triplet_extra_fields:
                resized.add_field(key, value, is_triplet=True)
            else:
                resized.add_field(key, value)
        return resized

    def transpose(self, method: int) -> "BoxListLite":
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError("Only horizontal and vertical flip are implemented")
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            to_remove = 1
            transposed = torch.cat(
                (
                    image_width - xmax - to_remove,
                    ymin,
                    image_width - xmin - to_remove,
                    ymax,
                ),
                dim=-1,
            )
        else:
            transposed = torch.cat(
                (
                    xmin,
                    image_height - ymax,
                    xmax,
                    image_height - ymin,
                ),
                dim=-1,
            )
        flipped = BoxListLite(transposed, self.size, mode="xyxy").convert(self.mode)
        for key, value in self.extra_fields.items():
            if hasattr(value, "transpose") and not isinstance(value, torch.Tensor):
                value = value.transpose(method)
            if key in self.triplet_extra_fields:
                flipped.add_field(key, value, is_triplet=True)
            else:
                flipped.add_field(key, value)
        return flipped

    def clip_to_image(self, remove_empty: bool = True) -> "BoxListLite":
        to_remove = 1
        clipped = self.copy()
        clipped.bbox[:, 0].clamp_(min=0, max=self.size[0] - to_remove)
        clipped.bbox[:, 1].clamp_(min=0, max=self.size[1] - to_remove)
        clipped.bbox[:, 2].clamp_(min=0, max=self.size[0] - to_remove)
        clipped.bbox[:, 3].clamp_(min=0, max=self.size[1] - to_remove)
        if not remove_empty:
            return clipped
        box = clipped.bbox
        keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
        return clipped[keep] if keep.any() else clipped

    def area(self) -> torch.Tensor:
        if self.mode == "xyxy":
            to_remove = 1
            return (self.bbox[:, 2] - self.bbox[:, 0] + to_remove) * (self.bbox[:, 3] - self.bbox[:, 1] + to_remove)
        return self.bbox[:, 2] * self.bbox[:, 3]

    def to(self, device: str | torch.device) -> "BoxListLite":
        copied_fields: dict[str, Any] = {}
        for name, value in self.extra_fields.items():
            copied_fields[name] = value.to(device) if hasattr(value, "to") else value
        return BoxListLite(
            bbox=self.bbox.to(device),
            size=self.size,
            mode=self.mode,
            extra_fields=copied_fields,
            triplet_extra_fields=list(self.triplet_extra_fields),
        )

    def copy(self) -> "BoxListLite":
        copied = BoxListLite(self.bbox.clone(), self.size, self.mode)
        copied._copy_extra_fields(self)
        return copied

    def copy_with_fields(self, fields: list[str] | tuple[str, ...] | str, skip_missing: bool = False) -> "BoxListLite":
        copied = BoxListLite(self.bbox.clone(), self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                copied.add_field(field, self.get_field(field), is_triplet=field in self.triplet_extra_fields)
            elif not skip_missing:
                raise KeyError("Field '{}' not found".format(field))
        return copied

    def __getitem__(self, item: Any) -> "BoxListLite":
        sliced = BoxListLite(self.bbox[item], self.size, self.mode)
        for key, value in self.extra_fields.items():
            if key in self.triplet_extra_fields and isinstance(value, torch.Tensor):
                sliced.add_field(key, value[item][:, item], is_triplet=True)
            elif isinstance(value, torch.Tensor):
                sliced.add_field(key, value[item], is_triplet=key in self.triplet_extra_fields)
            else:
                sliced.add_field(key, value, is_triplet=key in self.triplet_extra_fields)
        return sliced

    def __len__(self) -> int:
        return int(self.bbox.shape[0])

    def __repr__(self) -> str:
        return "BoxListLite(num_boxes={}, image_width={}, image_height={}, mode={})".format(
            len(self),
            self.size[0],
            self.size[1],
            self.mode,
        )
