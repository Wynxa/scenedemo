from __future__ import annotations

import argparse
from pathlib import Path

from services.hazard_reasoning_service.bootstrap_runtime import bootstrap_runtime

bootstrap_runtime()

from services.hazard_reasoning_service.app.io.json_io import dump_json, load_json
from services.hazard_reasoning_service.app.io.runtime_config_io import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone hazard reasoning inference from a scene graph JSON."
    )
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--scene-graph-json", type=Path, required=True)
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_scene_graph_item(scene_graph_path: Path, image_id: str | None) -> dict:
    payload = load_json(scene_graph_path)
    if isinstance(payload, dict):
        if "image_id" in payload:
            return payload
        raise ValueError("Scene graph JSON dict must contain 'image_id'.")

    if not isinstance(payload, list):
        raise ValueError("Scene graph JSON must be either a dict or a list of dicts.")

    if image_id is None:
        if len(payload) != 1:
            raise ValueError("Scene graph JSON contains multiple items; please provide --image-id.")
        return payload[0]

    for item in payload:
        if item.get("image_id") == image_id:
            return item
    raise ValueError(f"Image id '{image_id}' not found in {scene_graph_path}.")


def main() -> None:
    args = parse_args()

    from services.hazard_reasoning_service.app.services.hazard_inference_service import (
        HazardReasoningService,
    )

    runtime_config = load_runtime_config(args.runtime_config)
    scene_graph_item = load_scene_graph_item(args.scene_graph_json, args.image_id)
    service = HazardReasoningService(runtime_config)
    result = service.infer_from_scene_graph(scene_graph_item)
    dump_json(args.output_json, result)
    print(result["image_level_result"])
    print(f"Saved inference JSON to {args.output_json}")


if __name__ == "__main__":
    main()
