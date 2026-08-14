from __future__ import annotations

import argparse

from services.scene_graph_service.app.config import RuntimePathsConfig, ServiceOwnedRuntimeConfig
from services.scene_graph_service.app.io.prediction_io import save_prediction_bundle
from services.scene_graph_service.app.runner import OracleSceneGraphRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate service-owned prediction json files from GT/oracle data for end-to-end evaluator smoke tests."
    )
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--roidb-file", required=True)
    parser.add_argument("--dict-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-im", type=int, default=-1)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = ServiceOwnedRuntimeConfig(
        paths=RuntimePathsConfig(
            img_dir=args.img_dir,
            roidb_file=args.roidb_file,
            dict_file=args.dict_file,
            image_file=args.image_file,
        )
    )
    runner = OracleSceneGraphRunner(runtime_config)
    bundle = runner.predict_dataset(split=args.split, num_im=args.num_im)
    save_prediction_bundle(bundle, args.output_dir)
    print(f"Saved {len(bundle.predictions)} oracle prediction files to: {args.output_dir}")


if __name__ == "__main__":
    main()
