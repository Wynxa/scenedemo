from __future__ import annotations

import argparse
from pathlib import Path

from services.detector_service.bootstrap_runtime import bootstrap_runtime

bootstrap_runtime()

from services.detector_service.app.io.json_io import dump_json
from services.detector_service.app.io.runtime_config_io import load_runtime_config
from services.detector_service.app.services.detector_service import LayeredDetectorService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the layered detector service on a single uploaded image."
    )
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = load_runtime_config(args.runtime_config)
    service = LayeredDetectorService(runtime_config)
    result = service.infer_image(args.image_path, image_id=args.image_id)
    dump_json(args.output_json, result)
    print(f"Detected {len(result['objects'])} objects.")
    print(f"Saved detector JSON to {args.output_json}")


if __name__ == "__main__":
    main()
