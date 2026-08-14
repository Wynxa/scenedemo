from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from services.scene_graph_service.app.config import ServiceOwnedRuntimeConfig
from services.scene_graph_service.app.data import VGH5DatasetLite
from services.scene_graph_service.app.eval.service_eval_runner import (
    evaluate_predictions_with_service_evaluator,
)
from services.scene_graph_service.app.io.prediction_io import save_prediction_bundle
from services.scene_graph_service.app.runner.factory import build_runner


@dataclass
class ServiceOwnedEvalResult:
    report: str
    prediction_dir: str
    num_predictions: int


def run_service_owned_eval_pipeline(
    *,
    runtime_config: ServiceOwnedRuntimeConfig,
    split: str,
    runner_name: str,
    prediction_dir: str,
    num_im: int = -1,
    iou_thres: float = 0.5,
    eval_all_metrics: bool = True,
    print_detail: bool = True,
) -> ServiceOwnedEvalResult:
    runner = build_runner(runner_name, runtime_config)
    bundle = runner.predict_dataset(split=split, num_im=num_im)
    save_prediction_bundle(bundle, prediction_dir)

    eval_dataset = VGH5DatasetLite(
        img_dir=runtime_config.paths.img_dir,
        roidb_file=runtime_config.paths.roidb_file,
        dict_file=runtime_config.paths.dict_file,
        image_file=runtime_config.paths.image_file,
        split=split,
        num_im=num_im,
    )
    train_dataset = VGH5DatasetLite(
        img_dir=runtime_config.paths.img_dir,
        roidb_file=runtime_config.paths.roidb_file,
        dict_file=runtime_config.paths.dict_file,
        image_file=runtime_config.paths.image_file,
        split="train",
    )

    report, _ = evaluate_predictions_with_service_evaluator(
        eval_dataset=eval_dataset,
        prediction_dir=Path(prediction_dir),
        train_dataset=train_dataset,
        mode=runtime_config.mode,
        iou_thres=iou_thres,
        eval_all_metrics=eval_all_metrics,
        print_detail=print_detail,
    )
    return ServiceOwnedEvalResult(
        report=report,
        prediction_dir=str(Path(prediction_dir)),
        num_predictions=len(bundle.predictions),
    )
