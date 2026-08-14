from __future__ import annotations

import argparse
from pathlib import Path

from services.hazard_reasoning_service.bootstrap_runtime import bootstrap_runtime

bootstrap_runtime()

from services.hazard_reasoning_service.app.io.json_io import dump_json, load_json


LABEL_MAP = {
    "worker": "worker",
    "helmet": "helmet",
    "vest": "vest",
    "gloves": "gloves",
    "face_mask": "face_mask",
    "tools": "tools",
    "scaffold": "scaffold",
    "excavator": "excavator",
    "unprotected_edge": "unprotected_edge",
    "bricklaying_zone": "bricklaying_zone",
    "rebar_zone": "rebar_zone",
    "钢筋作业区": "rebar_zone",
    "砌砖作业区": "bricklaying_zone",
    "临边": "unprotected_edge",
    "脚手架": "scaffold",
    "工具": "tools",
    "挖掘机": "excavator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Label Studio extracted annotation JSON into a scene graph JSON."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-id", type=str, default=None)
    return parser.parse_args()


def percent_to_abs(value: float, total: float) -> float:
    return float(value) * float(total) / 100.0


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json)
    annotations = payload.get("annotations", [])
    if not annotations:
        raise ValueError("No annotations found in input JSON.")

    result_items = annotations[0].get("result", [])
    rectangles = [item for item in result_items if item.get("type") == "rectanglelabels"]
    relations = [item for item in result_items if item.get("type") == "relation"]
    if not rectangles:
        raise ValueError("No rectanglelabels found in input JSON.")

    id_to_object = {}
    objects = []

    for object_index, item in enumerate(rectangles):
        value = item["value"]
        label = value["rectanglelabels"][0]
        normalized_name = LABEL_MAP.get(label, label)
        width = float(item["original_width"])
        height = float(item["original_height"])
        x = percent_to_abs(value["x"], width)
        y = percent_to_abs(value["y"], height)
        w = percent_to_abs(value["width"], width)
        h = percent_to_abs(value["height"], height)
        object_row = {
            "object_index": object_index,
            "object_id": object_index + 1,
            "name": normalized_name,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "bbox_xyxy": [x, y, x + w, y + h],
            "source_label": label,
        }
        id_to_object[item["id"]] = object_row
        objects.append(object_row)

    relationships = []
    for relation_index, item in enumerate(relations):
        subject = id_to_object.get(item["from_id"])
        obj = id_to_object.get(item["to_id"])
        if subject is None or obj is None:
            continue
        predicate = item.get("labels", [None])[0]
        if not predicate:
            continue
        relationships.append(
            {
                "relationship_id": relation_index + 1,
                "subject_id": subject["object_id"],
                "object_id": obj["object_id"],
                "predicate": predicate,
            }
        )

    image_path = payload.get("data", {}).get("image", "")
    image_id = args.image_id or args.input_json.stem.replace("extracted_annotation_", "")
    scene_graph = {
        "image_id": image_id,
        "file_name": Path(image_path).name if image_path else f"{image_id}.jpg",
        "image_path": image_path,
        "objects": objects,
        "relationships": relationships,
        "meta": {
            "source": "label_studio_extracted_annotation",
            "input_json": str(args.input_json),
        },
    }
    dump_json(args.output_json, scene_graph)
    print(f"Saved scene graph JSON to {args.output_json}")


if __name__ == "__main__":
    main()
