from __future__ import annotations

import sys
import types
from pathlib import Path


def get_service_root() -> Path:
    return Path(__file__).resolve().parent


def bootstrap_runtime() -> Path:
    service_root = get_service_root()
    parent_dir = service_root.parent
    vendor_root = service_root / "vendor"

    for path in (str(parent_dir), str(vendor_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    services_module = sys.modules.get("services")
    if services_module is None:
        services_module = types.ModuleType("services")
        services_module.__path__ = [str(parent_dir)]
        sys.modules["services"] = services_module

    scene_graph_module = sys.modules.get("services.scene_graph_service")
    if scene_graph_module is None:
        scene_graph_module = types.ModuleType("services.scene_graph_service")
        scene_graph_module.__path__ = [str(service_root)]
        sys.modules["services.scene_graph_service"] = scene_graph_module
        setattr(services_module, "scene_graph_service", scene_graph_module)

    top_level_module = sys.modules.get("scene_graph_service")
    if top_level_module is None:
        top_level_module = types.ModuleType("scene_graph_service")
        top_level_module.__path__ = [str(service_root)]
        sys.modules["scene_graph_service"] = top_level_module

    return service_root
