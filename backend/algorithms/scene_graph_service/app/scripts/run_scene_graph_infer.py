from __future__ import annotations

import argparse
import os
from pathlib import Path

from services.scene_graph_service.app.adapters.htcl_runtime_adapter import LegacyHtclConfig
from services.scene_graph_service.app.io.json_io import load_json, save_json
from services.scene_graph_service.app.services.relation_inference_service import (
    RelationInferenceRequest,
    RelationInferenceService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scene graph inference service CLI.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--label-dict",
        default=str(
            Path(__file__).resolve().parents[4]
            / "datasets"
            / "construction_vg_raw_train_test_full_20260703"
            / "vg"
            / "VG-SGG-dicts-with-attri.json"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-rel-score", type=float, default=0.0)
    parser.add_argument("--topk-relations", type=int, default=0)
    parser.add_argument("opts", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json)
    request = RelationInferenceRequest(
        image_id=payload.get("image_id"),
        image_path=payload["image_path"],
        objects=payload.get("objects", payload.get("detections", [])),
    )

    runtime_config = LegacyHtclConfig(
        config_file=args.config_file,
        weight_file=args.weight_file,
        label_dict=args.label_dict,
        device=args.device,
        min_rel_score=args.min_rel_score,
        topk_relations=args.topk_relations,
        opts=args.opts,
    )
    service = RelationInferenceService(runtime_config)
    result = service.infer(request)
    save_json(args.output_json, result)

    print("Scene graph inference finished.")
    print("Objects: {}".format(len(result["objects"])))
    print("Relationships: {}".format(len(result["relationships"])))
    print("Saved to: {}".format(os.path.abspath(args.output_json)))


if __name__ == "__main__":
    main()
