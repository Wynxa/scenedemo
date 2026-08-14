from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from services.scene_graph_service.app.data import VGH5DatasetLite
from services.scene_graph_service.app.modules.relation_feature_extractors_lite import RelationFeatureExtractorLite
from services.scene_graph_service.app.services.request_preprocessor import build_target_boxlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test service-owned relation feature extractor.")
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--roidb-file", required=True)
    parser.add_argument("--dict-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-im", type=int, default=1)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = VGH5DatasetLite(
        img_dir=args.img_dir,
        roidb_file=args.roidb_file,
        dict_file=args.dict_file,
        image_file=args.image_file,
        split=args.split,
        num_im=args.num_im,
    )
    sample = dataset[0]
    label_to_idx = {name: idx for idx, name in enumerate(dataset.ind_to_classes)}
    input_objects = [
        {
            "object_id": idx,
            "label_id": int(label_id),
            "bbox": bbox,
            "score": 1.0,
        }
        for idx, (bbox, label_id) in enumerate(zip(sample.boxes_xyxy, sample.labels))
    ]
    target = build_target_boxlist(input_objects, (sample.width, sample.height), label_to_idx)
    rel_pair_idxs = [torch.as_tensor([[int(r[0]), int(r[1])] for r in sample.relationships], dtype=torch.int64)]

    extractor = RelationFeatureExtractorLite(
        in_channels=512,
        pooler_resolution=7,
        pooler_scales=(0.25, 0.125, 0.0625, 0.03125),
        pooler_sampling_ratio=2,
        mlp_head_dim=4096,
        pool_all_levels=True,
        separate_spatial=False,
    )
    feature_maps = [
        torch.randn(1, 512, 200, 150),
        torch.randn(1, 512, 100, 75),
        torch.randn(1, 512, 50, 38),
        torch.randn(1, 512, 25, 19),
    ]
    union_features = extractor(feature_maps, [target], rel_pair_idxs)
    summary = {
        "image_id": sample.image_id,
        "num_boxes": len(sample.boxes_xyxy),
        "num_relations": len(sample.relationships),
        "union_feature_shape": list(union_features.shape),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
