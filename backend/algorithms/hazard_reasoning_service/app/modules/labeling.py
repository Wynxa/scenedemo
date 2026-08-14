from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


RelationTuple = Tuple[str, str]


def build_object_lookup(objects: Iterable[dict]) -> Dict[int, dict]:
    return {int(obj["object_id"]): obj for obj in objects}


def build_worker_relations(objects: List[dict], relationships: List[dict]) -> Dict[int, List[dict]]:
    object_lookup = build_object_lookup(objects)
    worker_relations: Dict[int, List[dict]] = defaultdict(list)

    for rel in relationships:
        subject_id = int(rel["subject_id"])
        subject = object_lookup.get(subject_id)
        if not subject or subject["name"] != "worker":
            continue

        object_info = object_lookup.get(int(rel["object_id"]))
        if object_info is None:
            continue

        worker_relations[subject_id].append(
            {
                "predicate": rel["predicate"],
                "object_id": int(object_info["object_id"]),
                "object_name": object_info["name"],
            }
        )
    return worker_relations


def resolve_proximity_conflicts(relations: List[dict]) -> List[dict]:
    pair_to_predicates: Dict[Tuple[int, str], set] = defaultdict(set)
    relation_lookup: Dict[Tuple[int, str, str], dict] = {}

    for rel in relations:
        key = (int(rel["object_id"]), rel["object_name"])
        pair_to_predicates[key].add(rel["predicate"])
        relation_lookup[(int(rel["object_id"]), rel["object_name"], rel["predicate"])] = rel

    resolved: List[dict] = []
    for (object_id, object_name), predicates in pair_to_predicates.items():
        if "next_to" in predicates:
            resolved.append(relation_lookup[(object_id, object_name, "next_to")])
            continue
        if "near" in predicates:
            resolved.append(relation_lookup[(object_id, object_name, "near")])
            predicates = set(predicates)
            predicates.discard("near")
        for predicate in sorted(predicates):
            resolved.append(relation_lookup[(object_id, object_name, predicate)])

    resolved.sort(key=lambda x: (x["object_id"], x["predicate"], x["object_name"]))
    return resolved


def matches_context_policy(relation_pairs: List[RelationTuple], match_mode: str, trigger_relations: List[RelationTuple]) -> bool:
    if match_mode == "default":
        return True

    pair_set = set(relation_pairs)
    if match_mode == "any":
        return any(trigger in pair_set for trigger in trigger_relations)
    if match_mode == "all":
        return all(trigger in pair_set for trigger in trigger_relations)
    raise ValueError(f"Unsupported context match mode: {match_mode}")
