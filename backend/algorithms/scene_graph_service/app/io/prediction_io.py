from __future__ import annotations

import json
from pathlib import Path

from services.scene_graph_service.app.runner.base_runner import (
    SceneGraphPredictionBundle,
    SceneGraphPredictionResult,
)


def save_prediction_result(prediction: SceneGraphPredictionResult, output_path: str | Path) -> None:
    payload = {
        "meta": prediction.meta,
        "image_id": prediction.image_id,
        "image_path": prediction.image_path,
        "objects": prediction.objects,
        "relationships": prediction.relationships,
    }
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_prediction_bundle(bundle: SceneGraphPredictionBundle, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "meta": bundle.meta,
        "files": [],
    }
    for prediction in bundle.predictions:
        filename = f"{prediction.image_id}.json"
        save_prediction_result(prediction, output_dir / filename)
        manifest["files"].append(filename)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
