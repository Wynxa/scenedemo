from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVICE_ROOT_PARENT = Path(__file__).resolve().parents[3]
if str(SERVICE_ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT_PARENT))

from scene_graph_service.bootstrap_runtime import bootstrap_runtime

bootstrap_runtime()

from services.scene_graph_service.app.io.json_io import load_json, save_json
from services.scene_graph_service.app.io.runtime_config_io import load_runtime_config
from services.scene_graph_service.app.runner.htcl_service_runner import HTCLServiceRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image HTCL hybrid service inference from a request json.")
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = load_runtime_config(args.runtime_config)
    payload = load_json(args.input_json)
    runner = HTCLServiceRunner(runtime_config)
    result = runner.predict_request(
        image_id=payload.get("image_id"),
        image_path=payload["image_path"],
        objects=payload["objects"],
    )
    response = {
        "image_id": result.image_id,
        "image_path": result.image_path,
        "objects": result.objects,
        "relationships": result.relationships,
        "meta": result.meta,
    }
    save_json(args.output_json, response)
    print(json.dumps({"output_json": str(Path(args.output_json).resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
