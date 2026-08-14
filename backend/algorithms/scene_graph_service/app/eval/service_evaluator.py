from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.scene_graph_service.app.eval.sgg_metrics import (
    SGMeanRecall,
    SGNGMeanRecall,
    SGNGZeroShotRecall,
    SGNoGraphConstraintRecall,
    SGPairAccuracy,
    SGRecall,
    SGZeroShotRecall,
)


@dataclass
class ServiceSGGEvaluationConfig:
    mode: str = "predcls"
    iou_thres: float = 0.5
    num_rel_category: int = 8
    multiple_preds: bool = False
    attribute_on: bool = False
    num_attributes: int = 1
    eval_all_metrics: bool = True
    zeroshot_triplet: np.ndarray | None = None
    print_detail: bool = True


class ServiceSGGEvaluator:
    def __init__(self, config: ServiceSGGEvaluationConfig, predicate_names: list[str]) -> None:
        self.config = config
        self.predicate_names = predicate_names
        self.result_dict: dict[str, object] = {}
        self.evaluator = {}
        self._build_evaluators()

    def _build_evaluators(self) -> None:
        mode = self.config.mode
        eval_recall = SGRecall(self.result_dict)
        eval_recall.register_container(mode)
        self.evaluator["eval_recall"] = eval_recall

        eval_nog_recall = SGNoGraphConstraintRecall(self.result_dict)
        eval_nog_recall.register_container(mode)
        self.evaluator["eval_nog_recall"] = eval_nog_recall

        eval_zeroshot_recall = SGZeroShotRecall(self.result_dict)
        eval_zeroshot_recall.register_container(mode)
        self.evaluator["eval_zeroshot_recall"] = eval_zeroshot_recall

        eval_ng_zeroshot_recall = SGNGZeroShotRecall(self.result_dict)
        eval_ng_zeroshot_recall.register_container(mode)
        self.evaluator["eval_ng_zeroshot_recall"] = eval_ng_zeroshot_recall

        eval_pair_accuracy = SGPairAccuracy(self.result_dict)
        eval_pair_accuracy.register_container(mode)
        self.evaluator["eval_pair_accuracy"] = eval_pair_accuracy

        eval_mean_recall = SGMeanRecall(
            self.result_dict,
            self.config.num_rel_category,
            self.predicate_names,
            print_detail=self.config.print_detail,
        )
        eval_mean_recall.register_container(mode)
        self.evaluator["eval_mean_recall"] = eval_mean_recall

        eval_ng_mean_recall = SGNGMeanRecall(
            self.result_dict,
            self.config.num_rel_category,
            self.predicate_names,
            print_detail=self.config.print_detail,
        )
        eval_ng_mean_recall.register_container(mode)
        self.evaluator["eval_ng_mean_recall"] = eval_ng_mean_recall

    def build_global_container(self) -> dict:
        return {
            "zeroshot_triplet": self.config.zeroshot_triplet,
            "result_dict": self.result_dict,
            "mode": self.config.mode,
            "multiple_preds": self.config.multiple_preds,
            "num_rel_category": self.config.num_rel_category,
            "iou_thres": self.config.iou_thres,
            "attribute_on": self.config.attribute_on,
            "num_attributes": self.config.num_attributes,
        }

    def generate_report(self) -> str:
        mode = self.config.mode
        self.evaluator["eval_mean_recall"].calculate_mean_recall(mode)
        report = ""
        report += self.evaluator["eval_recall"].generate_print_string(mode)
        report += self.evaluator["eval_mean_recall"].generate_print_string(mode)

        if self.config.eval_all_metrics:
            self.evaluator["eval_ng_mean_recall"].calculate_mean_recall(mode)
            report += self.evaluator["eval_nog_recall"].generate_print_string(mode)
            report += self.evaluator["eval_zeroshot_recall"].generate_print_string(mode)
            report += self.evaluator["eval_ng_zeroshot_recall"].generate_print_string(mode)
            report += self.evaluator["eval_ng_mean_recall"].generate_print_string(mode)

        if self.config.mode in {"predcls", "sgcls"}:
            report += self.evaluator["eval_pair_accuracy"].generate_print_string(mode)
        return report
