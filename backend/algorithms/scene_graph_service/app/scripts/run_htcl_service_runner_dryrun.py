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

from services.scene_graph_service.app.config import RuntimePathsConfig, ServiceOwnedRuntimeConfig
from services.scene_graph_service.app.io.runtime_config_io import (
    dump_runtime_config,
    load_runtime_config,
)
from services.scene_graph_service.app.runner.htcl_service_runner import HTCLServiceRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run the service-owned HTCL runner without invoking legacy maskrcnn_benchmark runtime."
    )
    parser.add_argument("--runtime-config", default="")
    parser.add_argument("--img-dir", default="")
    parser.add_argument("--roidb-file", default="")
    parser.add_argument("--dict-file", default="")
    parser.add_argument("--image-file", default="")
    parser.add_argument("--weight-file", default="")
    parser.add_argument("--config-file", default="")
    parser.add_argument("--opts-config-file", default="")
    parser.add_argument("--glove-dir", default="")
    parser.add_argument("--conflict-groups-json", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="predcls", choices=["predcls", "sgcls", "sgdet"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-im", type=int, default=1)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def build_runtime_config_from_args(args: argparse.Namespace) -> ServiceOwnedRuntimeConfig:
    if args.runtime_config:
        return load_runtime_config(args.runtime_config)
    return ServiceOwnedRuntimeConfig(
        paths=RuntimePathsConfig(
            img_dir=args.img_dir,
            roidb_file=args.roidb_file,
            dict_file=args.dict_file,
            image_file=args.image_file,
            weight_file=args.weight_file,
            config_file=args.config_file,
            opts_config_file=args.opts_config_file,
            glove_dir=args.glove_dir,
            conflict_groups_json=args.conflict_groups_json,
        ),
        device=args.device,
        mode=args.mode,
    )


def main() -> None:
    args = parse_args()
    runtime_config = build_runtime_config_from_args(args)
    runner = HTCLServiceRunner(runtime_config)
    runtime_state = runner.describe_runtime()

    dataset_info = {
        "split": args.split,
        "num_im": args.num_im,
    }
    samples = []
    dataset = runner.predict_dataset if False else None
    preview_dataset = None
    from services.scene_graph_service.app.data import VGH5DatasetLite

    preview_dataset = VGH5DatasetLite(
        img_dir=runtime_config.paths.img_dir,
        roidb_file=runtime_config.paths.roidb_file,
        dict_file=runtime_config.paths.dict_file,
        image_file=runtime_config.paths.image_file,
        split=args.split,
        num_im=args.num_im,
    )
    for index in range(min(len(preview_dataset), args.num_im if args.num_im > 0 else 1)):
        sample = preview_dataset[index]
        prepared = runner.prepare_sample(sample)
        samples.append(
            {
                "image_id": sample.image_id,
                "image_path": sample.image_path,
                "num_boxes": len(sample.boxes_xyxy),
                "prepared_image_shape": list(prepared["prepared"].image_tensor.shape),
                "prepared_target_size": list(prepared["prepared"].resized_target.size),
            }
        )

    summary = {
        "runtime_config": dump_runtime_config(runtime_config),
        "runtime_state": runtime_state,
        "dataset_info": dataset_info,
        "preview_samples": samples,
        "next_step": "Port the actual HTCL relation forward core into HTCLServiceRunner._predict_prepared.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
