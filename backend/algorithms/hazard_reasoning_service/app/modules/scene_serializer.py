from __future__ import annotations

from typing import Dict, List

from services.hazard_reasoning_service.app.modules.context_config import (
    get_context_text,
    get_relation_type,
)


PPE_TO_TEXT = {
    "helmet": "helmet",
    "vest": "vest",
    "gloves": "gloves",
    "face_mask": "face mask",
}

OBJECT_TO_TEXT = {
    "worker": "worker",
    "excavator": "excavator",
    "rebar_zone": "rebar zone",
    "bricklaying_zone": "bricklaying zone",
    "unprotected_edge": "unprotected edge",
    "scaffold": "scaffold",
    "tools": "tools",
    "helmet": "helmet",
    "vest": "vest",
    "gloves": "gloves",
    "face_mask": "face mask",
}

PREDICATE_TO_TEXT = {
    "wearing": "wearing",
    "standing_on": "standing on",
    "operating": "operating",
    "using": "using",
    "near": "near",
    "next_to": "next to",
    "co_working_with": "co-working with",
}


def build_scene_text_compact(sample: Dict) -> str:
    evidence = " ; ".join(sample["core_relations"]) if sample["core_relations"] else "none"
    return f"worker [OBSERVED_RELATIONS] {evidence}"


def _to_text_list(values: List[str], value_map: Dict[str, str]) -> str:
    if not values:
        return "none"
    return " ".join(value_map.get(v, v) for v in values)


def build_scene_text_structured(sample: Dict, context_policy_path: str) -> str:
    evidence_parts = []
    for rel in sample.get("relations", []):
        pred = PREDICATE_TO_TEXT.get(rel["predicate"], rel["predicate"])
        obj = OBJECT_TO_TEXT.get(rel["object_name"], rel["object_name"])
        evidence_parts.append(f"{pred} {obj}")

    evidence_text = " ; ".join(evidence_parts) if evidence_parts else "none"
    observed_ppe = [
        rel["object_name"]
        for rel in sample.get("relations", [])
        if rel["predicate"] == "wearing"
    ]
    lines = [
        "[WORKER] worker",
        f"[OBSERVED_PPE] {_to_text_list(observed_ppe, PPE_TO_TEXT)}",
        f"[OBSERVED_RELATIONS] {evidence_text}",
    ]
    return "\n".join(lines)


def build_worker_output_record(sample: Dict, pred_label: str, pred_label_id: int, pred_score: float, topk: List[Dict], context_policy_path: str) -> Dict:
    relation_type = get_relation_type(sample["context_id"], context_policy_path)
    context_text = get_context_text(sample["context_id"], context_policy_path)
    return {
        "sample_id": sample["sample_id"],
        "worker_id": sample["worker_id"],
        "unsafe_behavior_category": pred_label,
        "worker_bbox_xyxy": sample["worker_bbox_xyxy"],
        "context_id": sample["context_id"],
        "context_text": context_text,
        "relation_type": relation_type,
        "required_ppe": sample["required_ppe"],
        "present_ppe": sample["present_ppe"],
        "missing_ppe": sample["missing_ppe"],
        "scene_text": sample["scene_text"],
        "scene_text_structured": sample["scene_text_structured"],
        "pred_major_label_id": pred_label_id,
        "pred_major_label": pred_label,
        "pred_score": pred_score,
        "topk": topk,
    }
