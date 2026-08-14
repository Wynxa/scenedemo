from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


BOX_SCALE = 1024


@dataclass
class DatasetPaths:
    img_dir: str
    roidb_file: str
    dict_file: str
    image_file: str


@dataclass
class VGH5Sample:
    index: int
    image_path: str
    image_id: str
    width: int
    height: int
    boxes_xyxy: list[list[float]]
    labels: list[int]
    attributes: list[list[int]]
    relationships: list[list[int]]


class VGH5DatasetLite:
    def __init__(
        self,
        *,
        img_dir: str,
        roidb_file: str,
        dict_file: str,
        image_file: str,
        split: str,
        num_im: int = -1,
        num_val_im: int = 0,
        filter_empty_rels: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train/val/test")
        self.paths = DatasetPaths(
            img_dir=img_dir,
            roidb_file=roidb_file,
            dict_file=dict_file,
            image_file=image_file,
        )
        self.split = split
        self.num_im = num_im
        self.num_val_im = num_val_im
        self.filter_empty_rels = filter_empty_rels

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = self._load_info(dict_file)
        (
            self.split_mask,
            self.gt_boxes,
            self.gt_classes,
            self.gt_attributes,
            self.relationships,
            self.image_ids,
        ) = self._load_graphs(roidb_file)
        self.filenames, self.img_info = self._load_image_filenames(img_dir, image_file)
        selected = np.where(self.split_mask)[0]
        missing_selected = [int(i) for i in selected if i >= len(self.filenames) or self.filenames[i] is None]
        if missing_selected:
            preview = ", ".join(str(i) for i in missing_selected[:20])
            raise FileNotFoundError(
                "Selected split contains images missing from image_data/disk alignment. "
                "Missing indices count={}, sample indices=[{}]. "
                "Please verify img_dir and image_data.json match the h5 dataset.".format(
                    len(missing_selected),
                    preview,
                )
            )
        self.filenames = [self.filenames[i] for i in selected]
        self.img_info = [self.img_info[i] for i in selected]
        self.image_ids = [self.image_ids[i] for i in range(len(self.image_ids))]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> VGH5Sample:
        info = self.img_info[index]
        boxes = self.gt_boxes[index].copy()
        width = int(info["width"])
        height = int(info["height"])
        boxes = boxes / BOX_SCALE * max(width, height)
        boxes = boxes.reshape(-1, 4).tolist()
        labels = self.gt_classes[index].astype(int).tolist()
        attributes = self.gt_attributes[index].astype(int).tolist()
        relationships = self.relationships[index].astype(int).tolist()
        raw_image_id = info.get("image_id", self.image_ids[index])
        return VGH5Sample(
            index=index,
            image_path=self.filenames[index],
            image_id=str(raw_image_id),
            width=width,
            height=height,
            boxes_xyxy=[[float(v) for v in row] for row in boxes],
            labels=labels,
            attributes=attributes,
            relationships=relationships,
        )

    @staticmethod
    def _load_info(dict_file: str):
        info = json.loads(Path(dict_file).read_text(encoding="utf-8"))
        info["label_to_idx"]["__background__"] = 0
        info["predicate_to_idx"]["__background__"] = 0
        info["attribute_to_idx"]["__background__"] = 0
        ind_to_classes = sorted(info["label_to_idx"], key=lambda k: info["label_to_idx"][k])
        ind_to_predicates = sorted(info["predicate_to_idx"], key=lambda k: info["predicate_to_idx"][k])
        ind_to_attributes = sorted(info["attribute_to_idx"], key=lambda k: info["attribute_to_idx"][k])
        return ind_to_classes, ind_to_predicates, ind_to_attributes

    def _load_graphs(self, roidb_file: str):
        with h5py.File(roidb_file, "r") as roi_h5:
            data_split = roi_h5["split"][:]
            split_flag = 2 if self.split == "test" else 0
            split_mask = data_split == split_flag
            split_mask &= roi_h5["img_to_first_box"][:] >= 0
            if self.filter_empty_rels:
                split_mask &= roi_h5["img_to_first_rel"][:] >= 0

            image_index = np.where(split_mask)[0]
            if self.num_im > -1:
                image_index = image_index[: self.num_im]
            if self.num_val_im > 0:
                if self.split == "val":
                    image_index = image_index[: self.num_val_im]
                elif self.split == "train":
                    image_index = image_index[self.num_val_im :]

            final_mask = np.zeros_like(data_split).astype(bool)
            final_mask[image_index] = True

            all_labels = roi_h5["labels"][:, 0]
            all_attributes = roi_h5["attributes"][:, :]
            all_boxes = roi_h5[f"boxes_{BOX_SCALE}"][:]
            all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
            all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

            im_to_first_box = roi_h5["img_to_first_box"][final_mask]
            im_to_last_box = roi_h5["img_to_last_box"][final_mask]
            im_to_first_rel = roi_h5["img_to_first_rel"][final_mask]
            im_to_last_rel = roi_h5["img_to_last_rel"][final_mask]
            all_relations = roi_h5["relationships"][:]
            all_relation_predicates = roi_h5["predicates"][:, 0]

            boxes = []
            gt_classes = []
            gt_attributes = []
            relationships = []
            kept_image_ids = []

            for i, image_id in enumerate(image_index):
                box_start = im_to_first_box[i]
                box_end = im_to_last_box[i]
                rel_start = im_to_first_rel[i]
                rel_end = im_to_last_rel[i]

                boxes_i = all_boxes[box_start : box_end + 1, :]
                gt_classes_i = all_labels[box_start : box_end + 1]
                gt_attributes_i = all_attributes[box_start : box_end + 1, :]

                if rel_start >= 0:
                    predicates = all_relation_predicates[rel_start : rel_end + 1]
                    obj_idx = all_relations[rel_start : rel_end + 1] - box_start
                    rels = np.column_stack((obj_idx, predicates))
                else:
                    rels = np.zeros((0, 3), dtype=np.int32)

                boxes.append(boxes_i)
                gt_classes.append(gt_classes_i)
                gt_attributes.append(gt_attributes_i)
                relationships.append(rels)
                kept_image_ids.append(str(image_id))

            return final_mask, boxes, gt_classes, gt_attributes, relationships, kept_image_ids

    @staticmethod
    def _load_image_filenames(img_dir: str, image_file: str):
        im_data = json.loads(Path(image_file).read_text(encoding="utf-8"))
        corrupted = {"1592.jpg", "1722.jpg", "4616.jpg", "4617.jpg"}
        filenames: list[str | None] = []
        img_info: list[dict[str, Any]] = []
        missing_count = 0
        for img in im_data:
            basename = "{}.jpg".format(img["image_id"])
            if basename in corrupted:
                filenames.append(None)
                img_info.append(img)
                continue
            filename = str(Path(img_dir) / basename)
            if not Path(filename).exists():
                missing_count += 1
            filenames.append(filename)
            img_info.append(img)
        if len(filenames) != len(img_info):
            raise RuntimeError("Image filename count does not match image info count.")
        if missing_count > 0:
            print(
                "Warning: {} images listed in image_data.json were not found under {}."
                .format(missing_count, img_dir)
            )
        return filenames, img_info

