from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import datasets as D
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from services.scene_graph_service.app.adapters.htcl_runtime_adapter import (
    HtclRuntimeAdapter,
    LegacyHtclConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scene_graph_service preprocessing against HTCL legacy relation metrics."
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--my-opts", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-dict", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("opts", nargs="*", default=None)
    return parser.parse_args()


def _resolve_dataset_names() -> tuple[str, ...]:
    if cfg.DATASETS.TO_TEST == "train":
        return cfg.DATASETS.TRAIN
    if cfg.DATASETS.TO_TEST == "val":
        return cfg.DATASETS.VAL
    return cfg.DATASETS.TEST


def _build_eval_datasets():
    paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
    dataset_catalog = paths_catalog.DatasetCatalog
    dataset_names = _resolve_dataset_names()
    datasets = []
    for dataset_name in dataset_names:
        data = dataset_catalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        if "capgraphs_file" in args:
            del args["capgraphs_file"]
        args["transforms"] = None
        datasets.append((dataset_name, factory(**args)))
    return datasets


def _target_to_objects(target) -> list[dict[str, object]]:
    boxes = target.convert("xyxy").bbox.tolist()
    labels = target.get_field("labels").tolist()
    objects = []
    for bbox, label_id in zip(boxes, labels):
        objects.append(
            {
                "bbox": [float(v) for v in bbox],
                "label_id": int(label_id),
            }
        )
    return objects


def main() -> None:
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    if args.my_opts:
        cfg.merge_from_file(args.my_opts)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    mkdir(cfg.OUTPUT_DIR)
    logger = setup_logger("scene_graph_service_eval", cfg.OUTPUT_DIR, 0)
    logger.info("Running legacy-compatible dataset evaluation with scene_graph_service preprocessing.")
    logger.info("Config file: %s", args.config_file)
    if args.my_opts:
        logger.info("Extra opts file: %s", args.my_opts)
    logger.info("Weight file: %s", cfg.MODEL.WEIGHT)
    logger.info("Device: %s", cfg.MODEL.DEVICE)
    logger.info("Dataset split selector: %s", cfg.DATASETS.TO_TEST)

    label_dict = args.label_dict
    if not label_dict:
        paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
        dataset_catalog = paths_catalog.DatasetCatalog
        dataset_name = _resolve_dataset_names()[0]
        data = dataset_catalog.get(dataset_name, cfg)
        label_dict = data["args"]["dict_file"]

    runtime_config = LegacyHtclConfig(
        config_file=args.config_file,
        weight_file=cfg.MODEL.WEIGHT,
        label_dict=label_dict,
        device=cfg.MODEL.DEVICE,
        opts=args.opts,
    )
    adapter = HtclRuntimeAdapter(runtime_config)

    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    for dataset_name, dataset in _build_eval_datasets():
        logger.info("Start compatibility evaluation on %s (%d images).", dataset_name, len(dataset))
        predictions = []
        for index in range(len(dataset)):
            image_path = dataset.filenames[index]
            target = dataset.get_groundtruth(index, evaluation=False)
            objects = _target_to_objects(target)
            result = adapter.infer_boxlist(image_path, objects)
            predictions.append(result)

            if (index + 1) % 50 == 0 or index + 1 == len(dataset):
                logger.info("Processed %d / %d images.", index + 1, len(dataset))

        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
        evaluate(
            cfg=cfg,
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            iou_types=iou_types,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
