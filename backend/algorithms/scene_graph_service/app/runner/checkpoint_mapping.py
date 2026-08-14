from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModuleMappingRule:
    name: str
    prefixes: list[str] = field(default_factory=list)
    notes: str = ""


DEFAULT_MAPPING_RULES = [
    ModuleMappingRule(
        name="backbone",
        prefixes=["backbone."],
        notes="Feature extractor trunk and FPN layers.",
    ),
    ModuleMappingRule(
        name="rpn",
        prefixes=["rpn."],
        notes="Region proposal network layers.",
    ),
    ModuleMappingRule(
        name="roi_box_head",
        prefixes=["roi_heads.box."],
        notes="ROI box feature extractor and predictor.",
    ),
    ModuleMappingRule(
        name="roi_relation_head",
        prefixes=["roi_heads.relation."],
        notes="Relation feature extractor, context encoder, and relation predictor.",
    ),
    ModuleMappingRule(
        name="roi_attribute_head",
        prefixes=["roi_heads.attribute."],
        notes="Optional attribute head layers.",
    ),
]


def map_checkpoint_keys_to_service_modules(
    state_dict_keys: list[str],
    mapping_rules: list[ModuleMappingRule] | None = None,
) -> dict[str, Any]:
    mapping_rules = mapping_rules or DEFAULT_MAPPING_RULES
    grouped: dict[str, list[str]] = defaultdict(list)
    unmatched: list[str] = []

    for key in state_dict_keys:
        matched = False
        for rule in mapping_rules:
            if any(key.startswith(prefix) for prefix in rule.prefixes):
                grouped[rule.name].append(key)
                matched = True
                break
        if not matched:
            unmatched.append(key)

    rules_payload = {
        rule.name: {
            "prefixes": rule.prefixes,
            "notes": rule.notes,
            "num_keys": len(grouped.get(rule.name, [])),
            "sample_keys": grouped.get(rule.name, [])[:10],
        }
        for rule in mapping_rules
    }
    return {
        "mapped_modules": rules_payload,
        "unmatched_count": len(unmatched),
        "unmatched_sample_keys": unmatched[:20],
    }
