import os
from collections import OrderedDict
from typing import Any


def build_scene_graph_payload(
    *,
    image_id: str,
    image_path: str,
    objects: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_name = os.path.basename(image_path)
    return OrderedDict(
        [
            ("meta", meta or {}),
            ("image_id", image_id),
            ("file_name", file_name),
            ("image_path", image_path),
            ("objects", objects),
            ("relationships", relationships),
            (
                "scene_graph",
                OrderedDict(
                    [
                        ("image_id", image_id),
                        ("file_name", file_name),
                        ("image_path", image_path),
                        ("objects", objects),
                        ("relationships", relationships),
                    ]
                ),
            ),
        ]
    )
