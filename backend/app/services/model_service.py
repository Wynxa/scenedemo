"""
Model service: model registry and training.

Wraps the existing train.py and eval_test.py scripts.
"""

import json
import os


def get_default_model_config():
    """Default ModelConfig matching the existing model_config.py."""
    return {
        "hiddenDim": 128,
        "numHeads": 4,
        "numLayers": 2,
        "dropout": 0.2,
        "sceneInputDim": 768,
        "numSceneNodeTypes": 16,
        "numBehaviorNodes": 32,
        "behaviorIdEmbDim": 128,
        "behaviorTextDim": 384,
        "numRiskNodes": 16,
        "riskIdEmbDim": 128,
        "riskTextDim": 384,
        "sceneEdgeGeomDim": 12,
        "bridgeEdgeFeatDim": 8,
        "knowledgeEdgeFeatDim": 8,
        "lossWeights": {
            "behavior": 1.0,
            "risk": 0.8,
            "consistency": 0.3,
            "softBehavior": 0.1,
            "candidate": 0.0,
        },
    }


def get_behavior_classes():
    """Return the list of unsafe behavior classes."""
    return [
        {"key": "climbing_scaffold_frame", "label": "攀爬脚手架框架", "category": "height_related"},
        {"key": "leaning_out", "label": "身体探出", "category": "height_related"},
        {"key": "standing_on_guardrail", "label": "站立在护栏上", "category": "height_related"},
        {"key": "throwing_material", "label": "抛掷物料", "category": "tool_related"},
        {"key": "leaning_outside_platform", "label": "平台外探身", "category": "height_related"},
        {"key": "crossing_guardrail", "label": "翻越护栏", "category": "height_related"},
        {"key": "working_outside_guardrail", "label": "护栏外作业", "category": "height_related"},
        {"key": "climbing_cross_brace", "label": "攀爬交叉支撑", "category": "height_related"},
        {"key": "unsafe_step_off", "label": "不安全下步", "category": "stability_related"},
        {"key": "missing_step", "label": "踏空", "category": "stability_related"},
        {"key": "unstable_posture", "label": "不稳定姿势", "category": "stability_related"},
        {"key": "throwing_objects", "label": "抛物行为", "category": "tool_related"},
    ]


def get_risk_mechanisms():
    """Return the list of risk mechanisms."""
    return [
        {"key": "fall_risk", "label": "坠落风险", "stages": ["precursor", "occurring", "critical"]},
        {"key": "struck_by_object_risk", "label": "物体打击风险", "stages": ["precursor", "occurring", "critical"]},
    ]


def compare_models(model_id_a, model_id_b):
    """Compare two model metrics side by side."""
    from app.models.model_registry import ModelRegistry
    from app import create_app

    app = create_app()
    with app.app_context():
        a = ModelRegistry.query.get(model_id_a)
        b = ModelRegistry.query.get(model_id_b)
        if not a or not b:
            return None

        metrics_a = json.loads(a.metrics_json) if a.metrics_json else {}
        metrics_b = json.loads(b.metrics_json) if b.metrics_json else {}

        return {
            "modelA": {"id": a.id, "name": a.name, "metrics": metrics_a},
            "modelB": {"id": b.id, "name": b.name, "metrics": metrics_b},
            "comparison": _compute_metric_diff(metrics_a, metrics_b),
        }


def _compute_metric_diff(a, b):
    """Compute difference between two metric dicts."""
    diff = {}
    all_keys = set(list(a.keys()) + list(b.keys()))
    for k in all_keys:
        if k in a and k in b:
            diff[k] = {
                "modelA": a[k],
                "modelB": b[k],
                "delta": round(float(b[k]) - float(a[k]), 4) if isinstance(a[k], (int, float)) else None,
            }
    return diff
