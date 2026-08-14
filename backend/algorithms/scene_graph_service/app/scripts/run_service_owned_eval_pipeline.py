from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.scene_graph_service.app.config import RuntimePathsConfig, ServiceOwnedRuntimeConfig
from services.scene_graph_service.app.services.service_owned_eval_pipeline import (
    run_service_owned_eval_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fully service-owned evaluation pipeline without maskrcnn_benchmark runtime dependencies."
    )
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--roidb-file", required=True)
    parser.add_argument("--dict-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--runner", default="oracle")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", default="predcls", choices=["predcls", "sgcls", "sgdet"])
    parser.add_argument("--num-im", type=int, default=-1)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--disable-all-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = ServiceOwnedRuntimeConfig(
        paths=RuntimePathsConfig(
            img_dir=args.img_dir,
            roidb_file=args.roidb_file,
            dict_file=args.dict_file,
            image_file=args.image_file,
        ),
        mode=args.mode,
    )
    result = run_service_owned_eval_pipeline(
        runtime_config=runtime_config,
        split=args.split,
        runner_name=args.runner,
        prediction_dir=args.prediction_dir,
        num_im=args.num_im,
        iou_thres=args.iou_thres,
        eval_all_metrics=not args.disable_all_metrics,
        print_detail=True,
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
                    "runner": args.runner,
                    "split": args.split,
                    "mode": args.mode,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
