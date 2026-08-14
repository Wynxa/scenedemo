from __future__ import annotations

from dataclasses import dataclass

from services.scene_graph_service.app.config import ServiceOwnedRuntimeConfig
from services.scene_graph_service.app.data import VGH5DatasetLite
from services.scene_graph_service.app.runner.base_runner import (
    SceneGraphPredictionBundle,
)
from services.scene_graph_service.app.runner.prediction_builder import build_prediction_result


@dataclass
class OracleSceneGraphRunner:
    runtime_config: ServiceOwnedRuntimeConfig

    def predict_dataset(self, *, split: str, num_im: int = -1) -> SceneGraphPredictionBundle:
        dataset = VGH5DatasetLite(
            img_dir=self.runtime_config.paths.img_dir,
            roidb_file=self.runtime_config.paths.roidb_file,
            dict_file=self.runtime_config.paths.dict_file,
            image_file=self.runtime_config.paths.image_file,
            split=split,
            num_im=num_im,
        )
        idx_to_label = {i: name for i, name in enumerate(dataset.ind_to_classes)}
        idx_to_predicate = {i: name for i, name in enumerate(dataset.ind_to_predicates)}

        prediction_rows: list[SceneGraphPredictionResult] = []
        for sample in dataset:
            input_objects = [
                {
                    "object_id": idx,
                    "label_id": int(label_id),
                    "bbox": bbox,
                    "score": 1.0,
                    "source": "oracle_gt",
                }
                for idx, (bbox, label_id) in enumerate(zip(sample.boxes_xyxy, sample.labels))
            ]
            rel_pair_idxs = [[int(rel[0]), int(rel[1])] for rel in sample.relationships]
            pred_rel_label_ids = [int(rel[2]) for rel in sample.relationships]
            pred_rel_scores = []
            for rel in sample.relationships:
                score_row = [0.0] * len(dataset.ind_to_predicates)
                score_row[int(rel[2])] = 1.0
                pred_rel_scores.append(score_row)
            prediction_rows.append(
                build_prediction_result(
                    image_id=sample.image_id,
                    image_path=sample.image_path,
                    input_objects=input_objects,
                    bboxes_xyxy=sample.boxes_xyxy,
                    pred_label_ids=[int(v) for v in sample.labels],
                    pred_scores=[1.0] * len(sample.labels),
                    rel_pair_idxs=rel_pair_idxs,
                    pred_rel_scores=pred_rel_scores,
                    pred_rel_label_ids=pred_rel_label_ids,
                    idx_to_label=idx_to_label,
                    idx_to_predicate=idx_to_predicate,
                    meta={
                        "runner": "oracle_gt",
                        "split": split,
                    },
                )
            )

        return SceneGraphPredictionBundle(
            predictions=prediction_rows,
            meta={
                "runner": "oracle_gt",
                "split": split,
                "num_images": len(prediction_rows),
            },
        )
