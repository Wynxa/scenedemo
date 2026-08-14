from __future__ import annotations

import sys
import types
from pathlib import Path


def get_service_root() -> Path:
    return Path(__file__).resolve().parent


def bootstrap_runtime() -> Path:
    service_root = get_service_root()
    parent_dir = service_root.parent

    for path in (str(parent_dir), str(service_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    services_module = sys.modules.get("services")
    if services_module is None:
        services_module = types.ModuleType("services")
        services_module.__path__ = [str(parent_dir)]
        sys.modules["services"] = services_module

    service_module = sys.modules.get("services.detector_service")
    if service_module is None:
        service_module = types.ModuleType("services.detector_service")
        service_module.__path__ = [str(service_root)]
        sys.modules["services.detector_service"] = service_module
        setattr(services_module, "detector_service", service_module)

    top_level_module = sys.modules.get("detector_service")
    if top_level_module is None:
        top_level_module = types.ModuleType("detector_service")
        top_level_module.__path__ = [str(service_root)]
        sys.modules["detector_service"] = top_level_module

    return service_root
