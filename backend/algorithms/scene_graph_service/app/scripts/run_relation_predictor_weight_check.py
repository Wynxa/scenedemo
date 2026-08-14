from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

SERVICE_ROOT_PARENT = Path(__file__).resolve().parents[3]
if str(SERVICE_ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT_PARENT))

from scene_graph_service.bootstrap_runtime import bootstrap_runtime

bootstrap_runtime()

from services.scene_graph_service.app.data import VGH5DatasetLite
from services.scene_graph_service.app.modules.relation_predictors_lite import (
    PENETHTCLLite,
    build_lite_htcl_cfg,
)
from services.scene_graph_service.app.services.request_preprocessor import build_target_boxlist


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether service-owned PENET_HTCL predictor can load the trained checkpoint.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--opts-config", required=True)
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--roidb-file", required=True)
    parser.add_argument("--dict-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--glove-dir", required=True)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_cfg = _deep_merge(_load_yaml(args.base_config), _load_yaml(args.opts_config))
    conflict_groups_json = (
        merged_cfg.get("MODEL", {})
        .get("CONFLICT_REENTRY", {})
        .get("GROUPS_JSON", "")
    )
    if conflict_groups_json and not Path(conflict_groups_json).is_absolute():
        conflict_groups_json = str((Path.cwd() / conflict_groups_json).resolve())

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
        conflict_groups_json=conflict_groups_json,
    )
    predictor = PENETHTCLLite(
        cfg=cfg,
        in_channels=4096,
        obj_classes=list(dataset.ind_to_classes),
        rel_classes=list(dataset.ind_to_predicates),
    )

    checkpoint = torch.load(args.weight_file, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    predictor_state_dict = {
        key[len("roi_heads.relation.predictor.") :]: value
        for key, value in state_dict.items()
        if key.startswith("roi_heads.relation.predictor.")
    }
    load_result = predictor.load_state_dict(predictor_state_dict, strict=False)

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
    rel_labels = [torch.as_tensor([int(r[2]) for r in sample.relationships], dtype=torch.int64)]

    roi_features = torch.randn(len(target), 4096)
    union_features = torch.randn(len(sample.relationships), 4096)
    predictor.eval()
    with torch.no_grad():
        entity_dists, relation_dists, _ = predictor(
            [target],
            rel_pair_idxs,
            rel_labels,
            None,
            roi_features,
            union_features,
        )

    summary = {
        "weight_file": args.weight_file,
        "num_predictor_keys_in_checkpoint": len(predictor_state_dict),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "first_image_id": sample.image_id,
        "num_boxes": len(target),
        "num_relations": len(sample.relationships),
        "entity_dists_shape": [list(t.shape) for t in entity_dists],
        "relation_dists_shape": [list(t.shape) for t in relation_dists],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
