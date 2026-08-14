from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.scene_graph_service.app.data import VGH5DatasetLite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Service-owned dataset inspection/evaluation entry without maskrcnn_benchmark dependencies."
    )
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--roidb-file", required=True)
    parser.add_argument("--dict-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-im", type=int, default=-1)
    parser.add_argument("--num-val-im", type=int, default=0)
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
        num_val_im=args.num_val_im,
    )

    total_boxes = sum(len(classes) for classes in dataset.gt_classes)
    total_relations = sum(len(rels) for rels in dataset.relationships)
    first = dataset[0] if len(dataset) > 0 else None
    summary = {
        "split": args.split,
        "num_images": len(dataset),
        "num_object_classes": len(dataset.ind_to_classes),
        "num_predicate_classes": len(dataset.ind_to_predicates),
        "total_boxes": int(total_boxes),
        "total_relations": int(total_relations),
        "first_sample": None
        if first is None
        else {
            "index": first.index,
            "image_id": first.image_id,
            "image_path": first.image_path,
            "width": first.width,
            "height": first.height,
            "num_boxes": len(first.boxes_xyxy),
            "num_relations": len(first.relationships),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Service-owned dataset loader finished without maskrcnn_benchmark imports.")
    print("Next step: connect this dataset reader to the service-owned model runner and evaluator.")


if __name__ == "__main__":
    main()
