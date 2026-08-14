from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce

import numpy as np

from services.scene_graph_service.app.eval.geometry import (
    argsort_desc,
    bbox_overlaps,
    compute_pred_matches,
    intersect_2d,
    rel_nms,
    triplet,
)


ind_head = [31, 20, 48, 30, 22, 29, 8, 50, 21, 1, 40, 49, 43, 38, 23, 41]
ind_middle = [6, 11, 7, 46, 33, 16, 25, 47, 19, 24, 14, 5, 9, 35, 44, 13, 10]
ind_tail = [4, 12, 36, 26, 32, 42, 45, 28, 2, 3, 17, 18, 34, 37, 27, 39, 15]


def _build_vg_frequency_summary(result_list_100: list[float], rel_name_list: list[str]):
    max_required_index = max(ind_head + ind_middle + ind_tail)
    if len(result_list_100) < max_required_index or len(rel_name_list) < max_required_index:
        return None
    values = np.array(result_list_100)
    detail_items = [(str(rel_name_list[i - 1]), float(result_list_100[i - 1])) for i in ind_head + ind_middle + ind_tail]
    return {
        "head": float(values[(np.array(ind_head) - 1)].mean()),
        "middle": float(values[(np.array(ind_middle) - 1)].mean()),
        "tail": float(values[(np.array(ind_tail) - 1)].mean()),
        "details": detail_items,
    }


def _safe_mean(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict: dict[str, object]) -> None:
        self.result_dict = result_dict

    @abstractmethod
    def register_container(self, mode: str) -> None:
        pass

    @abstractmethod
    def generate_print_string(self, mode: str) -> str:
        pass


class SGRecall(SceneGraphEvaluation):
    def register_container(self, mode: str) -> None:
        self.result_dict[mode + "_recall"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode: str) -> str:
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_recall"].items():
            result_str += "    R @ %d: %.4f; " % (k, _safe_mean(v))
        result_str += " for mode=%s, type=Recall(Main).\n" % mode
        return result_str

    def calculate_recall(self, global_container: dict, local_container: dict, mode: str) -> dict:
        pred_rel_inds = local_container["pred_rel_inds"]
        rel_scores = local_container["rel_scores"]
        gt_rels = local_container["gt_rels"]
        gt_classes = local_container["gt_classes"]
        gt_boxes = local_container["gt_boxes"]
        pred_classes = local_container["pred_classes"]
        pred_boxes = local_container["pred_boxes"]
        obj_scores = local_container["obj_scores"]
        iou_thres = global_container["iou_thres"]

        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)
        if mode in {"predcls", "sgcls"}:
            pred_rels_labels, pred_scores = rel_nms(pred_boxes, pred_classes, pred_rel_inds, rel_scores, 0.6)
            pred_rels = np.column_stack((pred_rel_inds, pred_rels_labels))
            sort_idx = np.argsort(-(pred_scores * obj_scores[pred_rel_inds[:, 0]] * obj_scores[pred_rel_inds[:, 1]]))
            pred_rels = pred_rels[sort_idx]
            pred_scores = pred_scores[sort_idx]

        gt_triplets, gt_triplet_boxes, _ = triplet(gt_rels, gt_classes, gt_boxes)
        local_container["gt_triplets"] = gt_triplets
        local_container["gt_triplet_boxes"] = gt_triplet_boxes
        pred_triplets, pred_triplet_boxes, _ = triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        pred_to_gt = compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode == "phrdet",
        )
        local_container["pred_to_gt"] = pred_to_gt
        for k in self.result_dict[mode + "_recall"]:
            match = reduce(np.union1d, pred_to_gt[:k])
            self.result_dict[mode + "_recall"][k].append(float(len(match)) / float(gt_rels.shape[0]))
        return local_container


class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def register_container(self, mode: str) -> None:
        self.result_dict[mode + "_recall_nogc"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode: str) -> str:
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_recall_nogc"].items():
            result_str += " ng-R @ %d: %.4f; " % (k, _safe_mean(v))
        result_str += " for mode=%s, type=No Graph Constraint Recall(Main).\n" % mode
        return result_str

    def calculate_recall(self, global_container: dict, local_container: dict, mode: str) -> dict:
        obj_scores = local_container["obj_scores"]
        pred_rel_inds = local_container["pred_rel_inds"]
        rel_scores = local_container["rel_scores"]
        pred_boxes = local_container["pred_boxes"]
        pred_classes = local_container["pred_classes"]
        gt_rels = local_container["gt_rels"]

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] + 1))
        nogc_pred_scores = rel_scores[nogc_score_inds[:, 0], nogc_score_inds[:, 1] + 1]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = triplet(
            nogc_pred_rels,
            pred_classes,
            pred_boxes,
            nogc_pred_scores,
            obj_scores,
        )
        gt_triplets = local_container["gt_triplets"]
        gt_triplet_boxes = local_container["gt_triplet_boxes"]
        iou_thres = global_container["iou_thres"]
        nogc_pred_to_gt = compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode == "phrdet",
        )
        local_container["nogc_pred_to_gt"] = nogc_pred_to_gt
        for k in self.result_dict[mode + "_recall_nogc"]:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            self.result_dict[mode + "_recall_nogc"][k].append(float(len(match)) / float(gt_rels.shape[0]))
        return local_container


class SGZeroShotRecall(SceneGraphEvaluation):
    def register_container(self, mode: str) -> None:
        self.result_dict[mode + "_zeroshot_recall"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode: str) -> str:
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_zeroshot_recall"].items():
            result_str += "   zR @ %d: %.4f; " % (k, _safe_mean(v))
        result_str += " for mode=%s, type=Zero Shot Recall.\n" % mode
        return result_str

    def prepare_zeroshot(self, global_container: dict, local_container: dict) -> None:
        gt_rels = local_container["gt_rels"]
        gt_classes = local_container["gt_classes"]
        zeroshot_triplets = global_container["zeroshot_triplet"]
        if zeroshot_triplets is None or len(zeroshot_triplets) == 0:
            self.zeroshot_idx = []
            return
        sub_id, obj_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[obj_id], pred_label))
        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def calculate_recall(self, global_container: dict, local_container: dict, mode: str) -> None:
        pred_to_gt = local_container["pred_to_gt"]
        for k in self.result_dict[mode + "_zeroshot_recall"]:
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                match_list = match.tolist() if not isinstance(match, (list, tuple)) else match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                self.result_dict[mode + "_zeroshot_recall"][k].append(float(zeroshot_match) / float(len(self.zeroshot_idx)))


class SGNGZeroShotRecall(SceneGraphEvaluation):
    def register_container(self, mode: str) -> None:
        self.result_dict[mode + "_ng_zeroshot_recall"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode: str) -> str:
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_ng_zeroshot_recall"].items():
            result_str += "ng-zR @ %d: %.4f; " % (k, _safe_mean(v))
        result_str += " for mode=%s, type=No Graph Constraint Zero Shot Recall.\n" % mode
        return result_str

    def prepare_zeroshot(self, global_container: dict, local_container: dict) -> None:
        gt_rels = local_container["gt_rels"]
        gt_classes = local_container["gt_classes"]
        zeroshot_triplets = global_container["zeroshot_triplet"]
        if zeroshot_triplets is None or len(zeroshot_triplets) == 0:
            self.zeroshot_idx = []
            return
        sub_id, obj_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[obj_id], pred_label))
        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def calculate_recall(self, global_container: dict, local_container: dict, mode: str) -> None:
        pred_to_gt = local_container["nogc_pred_to_gt"]
        for k in self.result_dict[mode + "_ng_zeroshot_recall"]:
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                match_list = match.tolist() if not isinstance(match, (list, tuple)) else match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                self.result_dict[mode + "_ng_zeroshot_recall"][k].append(float(zeroshot_match) / float(len(self.zeroshot_idx)))


class SGPairAccuracy(SceneGraphEvaluation):
    def register_container(self, mode: str) -> None:
        self.result_dict[mode + "_accuracy_hit"] = {20: [], 50: [], 100: []}
        self.result_dict[mode + "_accuracy_count"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode: str) -> str:
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_accuracy_hit"].items():
            if len(v) == 0 or len(self.result_dict[mode + "_accuracy_count"][k]) == 0:
                continue
            a_hit = _safe_mean(v)
            a_count = _safe_mean(self.result_dict[mode + "_accuracy_count"][k])
            result_str += "    A @ %d: %.4f; " % (k, a_hit / max(a_count, 1e-12))
        result_str += " for mode=%s, type=TopK Accuracy.\n" % mode
        return result_str

    def prepare_gtpair(self, local_container: dict) -> None:
        pred_pair_idx = local_container["pred_rel_inds"][:, 0] * 1024 + local_container["pred_rel_inds"][:, 1]
        gt_pair_idx = local_container["gt_rels"][:, 0] * 1024 + local_container["gt_rels"][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container: dict, local_container: dict, mode: str) -> None:
        pred_to_gt = local_container["pred_to_gt"]
        gt_rels = local_container["gt_rels"]
        for k in self.result_dict[mode + "_accuracy_hit"]:
            if mode != "sgdet":
                gt_pair_pred_to_gt = [p for p, flag in zip(pred_to_gt, self.pred_pair_in_gt) if flag]
                gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k]) if len(gt_pair_pred_to_gt) > 0 else []
                self.result_dict[mode + "_accuracy_hit"][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + "_accuracy_count"][k].append(float(gt_rels.shape[0]))


class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict: dict[str, object], num_rel: int, ind_to_predicates: list[str], print_detail: bool = False):
        super().__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]

    def register_container(self, mode: str) -> None:
        self.result_dict[mode + "_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + "_mean_recall_collect"] = {
            20: [[] for _ in range(self.num_rel)],
            50: [[] for _ in range(self.num_rel)],
            100: [[] for _ in range(self.num_rel)],
        }
        self.result_dict[mode + "_mean_recall_list"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode: str) -> str:
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_mean_recall"].items():
            result_str += "   mR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=Mean Recall.\n" % mode

        freq_summary = _build_vg_frequency_summary(self.result_dict[mode + "_mean_recall_list"][100], self.rel_name_list)
        if freq_summary is not None:
            result_str += "mR100: "
            result_str += " mR100_head: %.4f; " % freq_summary["head"]
            result_str += " mR100_middle: %.4f; " % freq_summary["middle"]
            result_str += " mR100_tail: %.4f; " % freq_summary["tail"]
            result_str += " for mode=%s, type=Mean Recall 100.\n" % mode
            if self.print_detail:
                result_str += "----------------------- Details ------------------------\n"
                for rel_name, rel_value in freq_summary["details"]:
                    result_str += "({}:{:.4f}) ".format(rel_name, rel_value)
                result_str += "\n--------------------------------------------------------\n"
        elif self.print_detail:
            result_str += "Custom predicate space detected; skip VG head/middle/tail breakdown.\n"
            result_str += "----------------------- Details ------------------------\n"
            for rel_name, rel_value in zip(self.rel_name_list, self.result_dict[mode + "_mean_recall_list"][100]):
                result_str += "({}:{:.4f}) ".format(str(rel_name), rel_value)
            result_str += "\n--------------------------------------------------------\n"
        return result_str

    def collect_mean_recall_items(self, global_container: dict, local_container: dict, mode: str) -> None:
        pred_to_gt = local_container["pred_to_gt"]
        gt_rels = local_container["gt_rels"]
        for k in self.result_dict[mode + "_mean_recall_collect"]:
            match = reduce(np.union1d, pred_to_gt[:k])
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1
            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + "_mean_recall_collect"][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self, mode: str) -> None:
        for k in self.result_dict[mode + "_mean_recall"]:
            sum_recall = 0.0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                collected = self.result_dict[mode + "_mean_recall_collect"][k][idx + 1]
                tmp_recall = 0.0 if len(collected) == 0 else np.mean(collected)
                self.result_dict[mode + "_mean_recall_list"][k].append(tmp_recall)
                sum_recall += tmp_recall
            self.result_dict[mode + "_mean_recall"][k] = sum_recall / float(num_rel_no_bg)


class SGNGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict: dict[str, object], num_rel: int, ind_to_predicates: list[str], print_detail: bool = False):
        super().__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]

    def register_container(self, mode: str) -> None:
        self.result_dict[mode + "_ng_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + "_ng_mean_recall_collect"] = {
            20: [[] for _ in range(self.num_rel)],
            50: [[] for _ in range(self.num_rel)],
            100: [[] for _ in range(self.num_rel)],
        }
        self.result_dict[mode + "_ng_mean_recall_list"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode: str) -> str:
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_ng_mean_recall"].items():
            result_str += "ng-mR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=No Graph Constraint Mean Recall.\n" % mode
        freq_summary = _build_vg_frequency_summary(self.result_dict[mode + "_ng_mean_recall_list"][100], self.rel_name_list)
        if freq_summary is not None:
            result_str += "ng-mR100:"
            result_str += "ng-mR100_head:%.4f; " % freq_summary["head"]
            result_str += "ng-mR100_middle:%.4f; " % freq_summary["middle"]
            result_str += "ng-mR100_tail:%.4f; " % freq_summary["tail"]
            result_str += " for mode=%s, type=No Graph Constraint Mean Recall 100.\n" % mode
            if self.print_detail:
                result_str += "----------------------- Details ------------------------\n"
                for rel_name, rel_value in freq_summary["details"]:
                    result_str += "({}:{:.4f}) ".format(rel_name, rel_value)
                result_str += "\n--------------------------------------------------------\n"
        elif self.print_detail:
            result_str += "Custom predicate space detected; skip VG head/middle/tail breakdown.\n"
            result_str += "----------------------- Details ------------------------\n"
            for rel_name, rel_value in zip(self.rel_name_list, self.result_dict[mode + "_ng_mean_recall_list"][100]):
                result_str += "({}:{:.4f}) ".format(str(rel_name), rel_value)
            result_str += "\n--------------------------------------------------------\n"
        return result_str

    def collect_mean_recall_items(self, global_container: dict, local_container: dict, mode: str) -> None:
        pred_to_gt = local_container["nogc_pred_to_gt"]
        gt_rels = local_container["gt_rels"]
        for k in self.result_dict[mode + "_ng_mean_recall_collect"]:
            match = reduce(np.union1d, pred_to_gt[:k])
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1
            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + "_ng_mean_recall_collect"][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self, mode: str) -> None:
        for k in self.result_dict[mode + "_ng_mean_recall"]:
            sum_recall = 0.0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                collected = self.result_dict[mode + "_ng_mean_recall_collect"][k][idx + 1]
                tmp_recall = 0.0 if len(collected) == 0 else np.mean(collected)
                self.result_dict[mode + "_ng_mean_recall_list"][k].append(tmp_recall)
                sum_recall += tmp_recall
            self.result_dict[mode + "_ng_mean_recall"][k] = sum_recall / float(num_rel_no_bg)
