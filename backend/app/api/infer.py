import json

from flask import Blueprint, current_app, request

from app.extensions import db
from app.models.inference import (
    DetectedObject,
    HazardImageSummary,
    HazardReasoningResult,
    ImageAsset,
    InferenceStageRun,
    InferenceTask,
    SceneGraphRelation,
)
from app.services.inference_service import InferenceOrchestrator
from app.services.inference_queue import inference_queue
from app.utils.auth import get_current_user_id, login_required
from app.utils.response import error, success, table_data

infer_bp = Blueprint("infer", __name__)


def _create_task_by_pipeline_type(pipeline_type: str, payload: dict):
    image_id = payload.get("imageId")
    if not image_id:
        return None, None, error("缺少 imageId", 400)

    ImageAsset.query.get_or_404(image_id)
    task, stage_runs = InferenceOrchestrator.create_inference_task(
        image_id=image_id,
        pipeline_type=pipeline_type,
        creator_id=get_current_user_id(),
        request_payload=payload,
    )
    return task, stage_runs, None


@infer_bp.route("/pipeline", methods=["POST"])
@login_required
def create_pipeline_task():
    data = request.get_json() or {}
    task, stage_runs, err = _create_task_by_pipeline_type("full_pipeline", data)
    if err:
        return err

    if bool(data.get("runNow", False)):
        if not inference_queue.submit(current_app._get_current_object(), task.id):
            return error("推理队列已满，请稍后重试", 429)
        return success({"task": task.to_dict(), "stageRuns": [row.to_dict() for row in stage_runs]}, msg="全流程推理任务已进入队列")

    return success({"task": task.to_dict(), "stageRuns": [row.to_dict() for row in stage_runs]}, msg="全流程推理任务已创建")


@infer_bp.route("/pipeline/run", methods=["POST"])
@login_required
def run_pipeline_sync():
    data = request.get_json() or {}
    task, stage_runs, err = _create_task_by_pipeline_type("full_pipeline", data)
    if err:
        return err
    if not inference_queue.submit(current_app._get_current_object(), task.id):
        return error("推理队列已满，请稍后重试", 429)
    return success({"task": task.to_dict(), "stageRuns": [row.to_dict() for row in stage_runs]}, msg="全流程推理任务已进入队列")


@infer_bp.route("/scene-graph", methods=["POST"])
@login_required
def create_scene_graph_task():
    data = request.get_json() or {}
    task, stage_runs, err = _create_task_by_pipeline_type("scene_graph_only", data)
    if err:
        return err
    return success({"task": task.to_dict(), "stageRuns": [row.to_dict() for row in stage_runs]}, msg="场景图任务已创建")


@infer_bp.route("/reasoning", methods=["POST"])
@login_required
def create_reasoning_task():
    data = request.get_json() or {}
    task, stage_runs, err = _create_task_by_pipeline_type("reasoning_only", data)
    if err:
        return err
    return success({"task": task.to_dict(), "stageRuns": [row.to_dict() for row in stage_runs]}, msg="危险推理任务已创建")


@infer_bp.route("/task/list", methods=["GET"])
@login_required
def list_tasks():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)
    status = request.args.get("status", "")
    pipeline_type = request.args.get("pipelineType", "")

    query = InferenceTask.query
    if status:
        query = query.filter_by(status=status)
    if pipeline_type:
        query = query.filter_by(pipeline_type=pipeline_type)
    query = query.order_by(InferenceTask.create_time.desc())

    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data([row.to_dict() for row in pagination.items], pagination.total)


@infer_bp.route("/task/<int:task_id>", methods=["GET"])
@login_required
def get_task(task_id):
    task = InferenceTask.query.get_or_404(task_id)
    image = ImageAsset.query.get(task.image_id) if task.image_id else None
    stage_runs = InferenceStageRun.query.filter_by(task_id=task.id).all()
    summary = HazardImageSummary.query.filter_by(task_id=task.id).order_by(HazardImageSummary.id.desc()).first()
    return success({"task": task.to_dict(), "image": image.to_dict() if image else None, "stageRuns": [row.to_dict() for row in stage_runs], "summary": summary.to_dict() if summary else None})


@infer_bp.route("/task/<int:task_id>/run", methods=["POST"])
@login_required
def run_existing_task(task_id):
    task = InferenceTask.query.get_or_404(task_id)
    if task.pipeline_type != "full_pipeline":
        return error("当前仅支持执行 full_pipeline 任务", 400)
    if task.status == "running":
        return error("任务正在执行", 409)
    if not inference_queue.submit(current_app._get_current_object(), task.id):
        return error("推理队列已满，请稍后重试", 429)
    return success({"task": task.to_dict()}, msg="任务已进入推理队列")


@infer_bp.route("/task/<int:task_id>/objects", methods=["GET"])
@login_required
def list_detected_objects(task_id):
    InferenceTask.query.get_or_404(task_id)
    rows = DetectedObject.query.filter_by(task_id=task_id).order_by(DetectedObject.object_index.asc()).all()
    return success([row.to_dict() for row in rows])


@infer_bp.route("/task/<int:task_id>/scene-graph", methods=["GET"])
@login_required
def get_scene_graph(task_id):
    InferenceTask.query.get_or_404(task_id)
    objects = DetectedObject.query.filter_by(task_id=task_id).order_by(DetectedObject.object_index.asc()).all()
    relationships = SceneGraphRelation.query.filter_by(task_id=task_id).order_by(SceneGraphRelation.id.asc()).all()
    return success({"objects": [row.to_dict() for row in objects], "relationships": [row.to_dict() for row in relationships]})


@infer_bp.route("/task/<int:task_id>/hazards", methods=["GET"])
@login_required
def get_hazards(task_id):
    InferenceTask.query.get_or_404(task_id)
    worker_results = HazardReasoningResult.query.filter_by(task_id=task_id).order_by(HazardReasoningResult.id.asc()).all()
    summary = HazardImageSummary.query.filter_by(task_id=task_id).order_by(HazardImageSummary.id.desc()).first()
    return success({"workerResults": [row.to_dict() for row in worker_results], "imageLevelResult": summary.to_dict() if summary else None})


@infer_bp.route("/task/<int:task_id>/stage/<stage_name>/mock-save", methods=["POST"])
@login_required
def mock_save_stage_output(task_id, stage_name):
    task = InferenceTask.query.get_or_404(task_id)
    stage_run = InferenceStageRun.query.filter_by(task_id=task.id, stage_name=stage_name).first()
    if not stage_run:
        return error("阶段不存在", 404)

    payload = request.get_json() or {}
    stage_run.status = payload.get("status", "completed")
    stage_run.output_json = json.dumps(payload, ensure_ascii=False)

    if stage_name == "detector":
        objects = payload.get("objects", [])
        DetectedObject.query.filter_by(task_id=task.id, stage_run_id=stage_run.id).delete()
        InferenceOrchestrator.save_detector_objects(task_id=task.id, stage_run_id=stage_run.id, image_id=task.image_id, objects=objects)
    elif stage_name == "scene_graph":
        relationships = payload.get("relationships", [])
        SceneGraphRelation.query.filter_by(task_id=task.id, stage_run_id=stage_run.id).delete()
        InferenceOrchestrator.save_scene_graph_relations(task_id=task.id, stage_run_id=stage_run.id, image_id=task.image_id, relationships=relationships)
    elif stage_name == "reasoning":
        worker_results = payload.get("worker_results", [])
        image_level_result = payload.get("image_level_result", {})
        HazardReasoningResult.query.filter_by(task_id=task.id, stage_run_id=stage_run.id).delete()
        HazardImageSummary.query.filter_by(task_id=task.id, image_id=task.image_id).delete()
        InferenceOrchestrator.save_hazard_results(task_id=task.id, stage_run_id=stage_run.id, image_id=task.image_id, worker_results=worker_results, image_level_result=image_level_result)

    db.session.commit()
    return success(stage_run.to_dict(), msg="阶段结果已写入数据库")
