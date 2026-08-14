from flask import Blueprint

from app.extensions import db
from app.models.dataset import Dataset
from app.models.inference import (
    DetectedObject,
    HazardImageSummary,
    HazardReasoningResult,
    ImageAsset,
    InferenceTask,
    SceneGraphRelation,
)
from app.models.model_registry import ModelRegistry
from app.utils.response import success
from app.utils.auth import login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/stats", methods=["GET"])
@login_required
def get_stats():
    """Dashboard stats from the scene-graph inference pipeline."""

    total_images = ImageAsset.query.count()
    total_inference_tasks = InferenceTask.query.count()
    completed_tasks = InferenceTask.query.filter_by(status="completed").count()
    running_tasks = InferenceTask.query.filter_by(status="running").count()
    failed_tasks = InferenceTask.query.filter_by(status="failed").count()

    total_objects = DetectedObject.query.count()
    total_relations = SceneGraphRelation.query.count()
    total_hazard_results = HazardReasoningResult.query.count()
    total_summaries = HazardImageSummary.query.count()
    active_models = ModelRegistry.query.filter_by(status="active").count()

    # Entity category distribution
    from sqlalchemy import func
    entity_rows = (
        db.session.query(
            DetectedObject.label_name,
            func.count(DetectedObject.id).label("cnt"),
        )
        .group_by(DetectedObject.label_name)
        .order_by(func.count(DetectedObject.id).desc())
        .all()
    )
    entity_dist = [{"name": r[0] or "unknown", "value": r[1]} for r in entity_rows]

    # Predicate category distribution
    pred_rows = (
        db.session.query(
            SceneGraphRelation.predicate_name,
            func.count(SceneGraphRelation.id).label("cnt"),
        )
        .group_by(SceneGraphRelation.predicate_name)
        .order_by(func.count(SceneGraphRelation.id).desc())
        .all()
    )
    predicate_dist = [{"name": r[0] or "unknown", "value": r[1]} for r in pred_rows]

    # Unsafe behavior distribution
    hazard_rows = (
        db.session.query(
            HazardReasoningResult.unsafe_behavior_category,
            func.count(HazardReasoningResult.id).label("cnt"),
        )
        .filter(HazardReasoningResult.pred_score > 0.5)
        .group_by(HazardReasoningResult.unsafe_behavior_category)
        .order_by(func.count(HazardReasoningResult.id).desc())
        .all()
    )
    hazard_dist = [{"name": r[0] or "safe", "value": r[1]} for r in hazard_rows]

    # Image-level hazard summary
    unsafe_images = HazardImageSummary.query.filter_by(has_unsafe_behavior=True).count()
    safe_images = total_summaries - unsafe_images

    # Recent inference tasks
    recent_tasks = (
        InferenceTask.query
        .order_by(InferenceTask.create_time.desc())
        .limit(8)
        .all()
    )

    return success({
        "totalImages": total_images,
        "totalInferenceTasks": total_inference_tasks,
        "completedTasks": completed_tasks,
        "runningTasks": running_tasks,
        "failedTasks": failed_tasks,
        "totalObjects": total_objects,
        "totalRelations": total_relations,
        "totalHazardResults": total_hazard_results,
        "totalSummaries": total_summaries,
        "unsafeImages": unsafe_images,
        "safeImages": safe_images,
        "activeModels": active_models,
        "entityDistribution": entity_dist,
        "predicateDistribution": predicate_dist,
        "hazardDistribution": hazard_dist,
        "recentTasks": [t.to_dict() for t in recent_tasks],
    })


@dashboard_bp.route("/charts/behavior-dist", methods=["GET"])
@login_required
def behavior_distribution():
    from sqlalchemy import func

    rows = (
        db.session.query(
            HazardReasoningResult.unsafe_behavior_category,
            func.count(HazardReasoningResult.id).label("cnt"),
        )
        .filter(HazardReasoningResult.pred_score > 0.5)
        .group_by(HazardReasoningResult.unsafe_behavior_category)
        .all()
    )

    categories = [
        "general_work", "rebar_work", "mechanical_operation",
        "excavation_proximity", "scaffold_work", "safe",
    ]

    return success({
        "categories": categories,
        "values": [
            sum(r[1] for r in rows if r[0] == c) for c in categories
        ],
    })


@dashboard_bp.route("/charts/entity-dist", methods=["GET"])
@login_required
def entity_distribution():
    from sqlalchemy import func

    rows = (
        db.session.query(
            DetectedObject.label_name,
            func.count(DetectedObject.id).label("cnt"),
        )
        .group_by(DetectedObject.label_name)
        .all()
    )

    entities = ["worker", "helmet", "excavator", "vest", "rebar_zone", "gloves", "scaffold"]

    return success({
        "categories": entities,
        "values": [
            sum(r[1] for r in rows if r[0] and r[0].lower() == e) for e in entities
        ],
    })
