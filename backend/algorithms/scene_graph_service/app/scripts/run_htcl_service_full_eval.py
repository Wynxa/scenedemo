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

from services.scene_graph_service.app.io.runtime_config_io import load_runtime_config
from services.scene_graph_service.app.services.service_owned_eval_pipeline import (
    run_service_owned_eval_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-dataset HTCL hybrid evaluation with service-owned relation head."
    )
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-im", type=int, default=-1)
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--disable-all-metrics", action="store_true")
    parser.add_argument("--quiet-detail", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = load_runtime_config(args.runtime_config)
    service_root = Path(__file__).resolve().parents[2]
    prediction_dir = (
        Path(args.prediction_dir)
        if args.prediction_dir
        else service_root / "tmp" / f"predictions_{args.split}_htcl_service"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)

    result = run_service_owned_eval_pipeline(
        runtime_config=runtime_config,
        split=args.split,
        runner_name="htcl_service",
        prediction_dir=str(prediction_dir),
        num_im=args.num_im,
        iou_thres=args.iou_thres,
        eval_all_metrics=not args.disable_all_metrics,
        print_detail=not args.quiet_detail,
    )

    print("=" * 100)
    print(result.report.rstrip())
    print("=" * 100)
    print("Prediction dir:", result.prediction_dir)
    print("Num predictions:", result.num_predictions)

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(
                {
                    "report": result.report,
                    "prediction_dir": result.prediction_dir,
                    "num_predictions": result.num_predictions,
                    "runner": "htcl_service",
                    "split": args.split,
                    "mode": runtime_config.mode,
                    "runtime_config": args.runtime_config,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
