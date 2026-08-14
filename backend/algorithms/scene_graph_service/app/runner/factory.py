from __future__ import annotations

from services.scene_graph_service.app.config import ServiceOwnedRuntimeConfig
from services.scene_graph_service.app.runner.base_runner import SceneGraphRunner
from services.scene_graph_service.app.runner.htcl_service_runner import HTCLServiceRunner
from services.scene_graph_service.app.runner.oracle_runner import OracleSceneGraphRunner


def build_runner(runner_name: str, runtime_config: ServiceOwnedRuntimeConfig) -> SceneGraphRunner:
    normalized = runner_name.strip().lower()
    if normalized == "oracle":
        return OracleSceneGraphRunner(runtime_config)
    if normalized == "htcl_service":
        return HTCLServiceRunner(runtime_config)
    raise ValueError(
        "Unsupported runner '{}'. Currently available runners: oracle, htcl_service".format(runner_name)
    )
