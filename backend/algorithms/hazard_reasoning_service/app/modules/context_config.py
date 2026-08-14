from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


RELATION_TYPE_TO_TEXT = {
    "routine_work": "routine work",
    "work_context": "work context",
    "tool_operation": "tool operation",
    "mechanical_operation": "mechanical operation",
    "hazardous_proximity": "hazardous proximity",
}


@lru_cache(maxsize=None)
def load_context_config(path: str | Path) -> Dict:
    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    context_entries = payload.get("contexts", [])
    context_to_text = {}
    context_to_relation_type = {}
    context_to_required_ppe = {}

    for item in context_entries:
        context_id = item["context_id"]
        context_to_text[context_id] = item.get("context_text", context_id.replace("_", " "))
        context_to_relation_type[context_id] = item.get("relation_type", "work_context")
        context_to_required_ppe[context_id] = list(item.get("required_ppe", []))

    return {
        "path": str(policy_path),
        "meta": payload.get("meta", {}),
        "contexts": context_entries,
        "default_safe_label": payload.get("default_safe_label", "safe"),
        "context_to_text": context_to_text,
        "context_to_relation_type": context_to_relation_type,
        "context_to_required_ppe": context_to_required_ppe,
        "relation_type_to_text": dict(RELATION_TYPE_TO_TEXT),
    }


def get_context_text(context_id: str, path: str | Path) -> str:
    config = load_context_config(path)
    return config["context_to_text"].get(context_id, context_id.replace("_", " "))


def get_relation_type(context_id: str, path: str | Path) -> str:
    config = load_context_config(path)
    return config["context_to_relation_type"].get(context_id, "work_context")


def get_required_ppe_map(path: str | Path) -> Dict[str, List[str]]:
    config = load_context_config(path)
    return dict(config["context_to_required_ppe"])
