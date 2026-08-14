from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVICE_ROOT_PARENT = Path(__file__).resolve().parents[3]
if str(SERVICE_ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT_PARENT))

from scene_graph_service.bootstrap_runtime import bootstrap_runtime

bootstrap_runtime()

from services.scene_graph_service.app.adapters.htcl_runtime_adapter import LegacyHtclConfig
from services.scene_graph_service.app.io.json_io import load_json, save_json
from services.scene_graph_service.app.services.detector_payload_adapter import DetectorPayloadAdapter
from services.scene_graph_service.app.services.relation_inference_service import (
    RelationInferenceRequest,
    RelationInferenceService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call scene graph service from detector-service style payload.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label-dict", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-rel-score", type=float, default=0.0)
    parser.add_argument("--topk-relations", type=int, default=0)
    parser.add_argument("opts", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json)
    adapter = DetectorPayloadAdapter()
    request_payload = adapter.build_relation_request(payload)

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
    result = service.infer(
        RelationInferenceRequest(
            image_id=request_payload.get("image_id"),
            image_path=request_payload["image_path"],
            objects=request_payload["objects"],
        )
    )
    detector_meta = request_payload.get("meta", {})
    result.setdefault("meta", {})
    result["meta"]["upstream_detector"] = detector_meta
    save_json(args.output_json, result)
    print("Scene graph inference from detector payload finished.")
    print("Saved to: {}".format(str(Path(args.output_json).resolve())))


if __name__ == "__main__":
    main()
