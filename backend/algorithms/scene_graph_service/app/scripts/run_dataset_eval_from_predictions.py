from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.scene_graph_service.app.data import VGH5DatasetLite
from services.scene_graph_service.app.eval.service_eval_runner import (
    evaluate_predictions_with_service_evaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scene_graph_service prediction json files with the service-owned evaluator."
    )
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--roidb-file", required=True)
    parser.add_argument("--dict-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", default="predcls", choices=["predcls", "sgcls", "sgdet"])
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--disable-all-metrics", action="store_true")
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dataset = VGH5DatasetLite(
        img_dir=args.img_dir,
        roidb_file=args.roidb_file,
        dict_file=args.dict_file,
        image_file=args.image_file,
        split=args.split,
    )
    train_dataset = VGH5DatasetLite(
        img_dir=args.img_dir,
        roidb_file=args.roidb_file,
        dict_file=args.dict_file,
        image_file=args.image_file,
        split="train",
    )

    report, result_dict = evaluate_predictions_with_service_evaluator(
        eval_dataset=eval_dataset,
        prediction_dir=args.prediction_dir,
        train_dataset=train_dataset,
        mode=args.mode,
        iou_thres=args.iou_thres,
        eval_all_metrics=not args.disable_all_metrics,
        print_detail=True,
    )

    print("=" * 100)
    print(report.rstrip())
    print("=" * 100)

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(
                {
                    "report": report,
                    "result_dict": _to_jsonable(result_dict),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
