from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.scene_graph_service.app.runner.checkpoint_inspector import inspect_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a relation checkpoint from scene_graph_service without maskrcnn_benchmark runtime imports."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_checkpoint(args.checkpoint, map_location="cpu")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
