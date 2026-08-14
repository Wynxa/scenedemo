from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SERVICE_ROOT_PARENT = Path(__file__).resolve().parents[3]
if str(SERVICE_ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT_PARENT))

from scene_graph_service.bootstrap_runtime import bootstrap_runtime

bootstrap_runtime()

from services.scene_graph_service.app.data import VGH5DatasetLite
from services.scene_graph_service.app.modules.relation_head_lite import (
    RelationHeadLite,
    load_relation_head_from_checkpoint,
)
from services.scene_graph_service.app.modules.relation_predictors_lite import (
    PENETHTCLLite,
    build_lite_htcl_cfg,
)
from services.scene_graph_service.app.services.request_preprocessor import build_target_boxlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check service-owned relation head weight loading and forward path.")
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--roidb-file", required=True)
    parser.add_argument("--dict-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--glove-dir", required=True)
    parser.add_argument("--conflict-groups-json", default="")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = VGH5DatasetLite(
        img_dir=args.img_dir,
        roidb_file=args.roidb_file,
        dict_file=args.dict_file,
        image_file=args.image_file,
        split="test",
        num_im=1,
    )
    cfg = build_lite_htcl_cfg(
        glove_dir=args.glove_dir,
        num_obj_classes=len(dataset.ind_to_classes),
        num_rel_classes=len(dataset.ind_to_predicates),
        conflict_groups_json=args.conflict_groups_json,
    )
    predictor = PENETHTCLLite(
        cfg=cfg,
        in_channels=4096,
        obj_classes=list(dataset.ind_to_classes),
        rel_classes=list(dataset.ind_to_predicates),
    )
    relation_head = RelationHeadLite(predictor=predictor)

    checkpoint = torch.load(args.weight_file, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    load_summary = load_relation_head_from_checkpoint(relation_head, state_dict)

    sample = dataset[0]
    label_to_idx = {name: idx for idx, name in enumerate(dataset.ind_to_classes)}
    input_objects = [
        {"object_id": idx, "label_id": int(label_id), "bbox": bbox, "score": 1.0}
        for idx, (bbox, label_id) in enumerate(zip(sample.boxes_xyxy, sample.labels))
    ]
    target = build_target_boxlist(input_objects, (sample.width, sample.height), label_to_idx)
    rel_pair_idxs = [torch.as_tensor([[int(r[0]), int(r[1])] for r in sample.relationships], dtype=torch.int64)]
    rel_labels = [torch.as_tensor([int(r[2]) for r in sample.relationships], dtype=torch.int64)]
    feature_maps = [
        torch.randn(1, 512, 200, 150),
        torch.randn(1, 512, 100, 75),
        torch.randn(1, 512, 50, 38),
        torch.randn(1, 512, 25, 19),
    ]

    relation_head.eval()
    with torch.no_grad():
        output = relation_head(
            features=feature_maps,
            proposals=[target],
            rel_pair_idxs=rel_pair_idxs,
            rel_labels=rel_labels,
        )

    summary = {
        "weight_file": args.weight_file,
        "load_summary": load_summary,
        "first_image_id": sample.image_id,
        "num_boxes": len(target),
        "num_relations": len(sample.relationships),
        "obj_logits_shape": [list(t.shape) for t in output.obj_logits],
        "rel_logits_shape": [list(t.shape) for t in output.rel_logits],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
