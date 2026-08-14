import json

from flask import Blueprint, request
from flask_socketio import emit

from app.extensions import db, socketio
from app.models.inference import (
    DetectedObject,
    HazardReasoningResult,
    ImageAsset,
    InferenceTask,
)
from app.services.inference_service import InferenceOrchestrator
from app.utils.response import success, error, table_data
from app.utils.auth import login_required, get_current_user_id

detection_bp = Blueprint("detection", __name__)

# Behavior label mapping (unified: detection + inference categories)
BEHAVIOR_LABEL_MAP = {
    "climbing_scaffold_frame": "攀爬脚手架框架",
    "leaning_out": "身体探出",
    "standing_on_guardrail": "站立在护栏上",
    "throwing_material": "抛掷物料",
    "leaning_outside_platform": "平台外探身",
    "crossing_guardrail": "翻越护栏",
    "working_outside_guardrail": "护栏外作业",
    "climbing_cross_brace": "攀爬交叉支撑",
    "unsafe_step_off": "不安全下步",
    "missing_step": "踏空",
    "unstable_posture": "不稳定姿势",
    "throwing_objects": "抛物行为",
    "general_work": "一般作业",
    "rebar_work": "钢筋作业",
    "mechanical_operation": "机械操作",
    "excavation_proximity": "开挖临近",
    "scaffold_work": "脚手架作业",
    "safe": "安全",
    "unknown": "未知",
}


def _behavior_label(category):
    return BEHAVIOR_LABEL_MAP.get(category, category or "未知")


def _hazard_to_result_row(h):
    """Convert a HazardReasoningResult to the format the frontend result list expects."""
    return {
        "id": h.id,
        "imageId": h.image_id,
        "primaryBehavior": h.unsafe_behavior_category or "unknown",
        "primaryRisk": h.pred_major_label or "",
        "riskSeverity": round(h.pred_score, 4) if h.pred_score else 0.0,
        "confidence": round(h.pred_score, 4) if h.pred_score else 0.0,
    }


def _hazard_to_detail_detection(h, obj):
    """Convert HazardReasoningResult + DetectedObject to frontend detection detail format."""
    # Bbox: DetectedObject stores x1,y1,x2,y2; frontend expects "x,y,w,h" string
    if obj:
        w = max(obj.bbox_x2 - obj.bbox_x1, 0)
        h_val = max(obj.bbox_y2 - obj.bbox_y1, 0)
        worker_bbox = f"{int(obj.bbox_x1)},{int(obj.bbox_y1)},{int(w)},{int(h_val)}"
    else:
        worker_bbox = "0,0,0,0"

    # Behavior probs from topk_json
    behavior_probs = {}
    topk_raw = json.loads(h.topk_json) if h.topk_json else []
    if topk_raw:
        if isinstance(topk_raw, list):
            for item in topk_raw:
                if isinstance(item, dict):
                    behavior_probs[item.get("label", item.get("name", ""))] = item.get("score", item.get("prob", 0))
                else:
                    behavior_probs[str(item)] = h.pred_score
        elif isinstance(topk_raw, dict):
            behavior_probs = topk_raw
    if not behavior_probs:
        behavior_probs = {h.unsafe_behavior_category or "unknown": round(h.pred_score, 4)}

    # Risk probs from PPE fields
    risk_probs = {}
    required_ppe = json.loads(h.required_ppe_json) if h.required_ppe_json else []
    present_ppe = json.loads(h.present_ppe_json) if h.present_ppe_json else []
    missing_ppe = json.loads(h.missing_ppe_json) if h.missing_ppe_json else []
    if missing_ppe:
        risk_probs["fall_risk"] = min(len(missing_ppe) / max(len(required_ppe), 1), 1.0)
    if required_ppe:
        risk_probs["struck_by_object_risk"] = round(1.0 - len(present_ppe) / max(len(required_ppe), 1), 4)
    if not risk_probs:
        risk_probs["fall_risk"] = round(h.pred_score, 4) if h.unsafe_behavior_category not in ("safe", "unknown") else 0.0

    return {
        "workerBbox": worker_bbox,
        "behaviorProbs": behavior_probs,
        "riskProbs": risk_probs,
        "riskSeverity": round(h.pred_score, 4) if h.pred_score else 0.0,
        "primaryBehavior": h.unsafe_behavior_category or "unknown",
        "primaryBehaviorLabel": _behavior_label(h.unsafe_behavior_category),
        "confidence": round(h.pred_score, 4) if h.pred_score else 0.0,
        "primaryRisk": h.pred_major_label or "",
    }


# ── Task endpoints (now backed by InferenceTask) ──────────────────────────

@detection_bp.route("/task", methods=["POST"])
@login_required
def create_task():
    data = request.get_json()
    image_id = data.get("imageId")
    if not image_id:
        return error("缺少 imageId", 400)

    # Validate image exists
    ImageAsset.query.get_or_404(image_id)

    pipeline_type = data.get("pipelineType", "full_pipeline")

    task, stage_runs = InferenceOrchestrator.create_inference_task(
        image_id=image_id,
        pipeline_type=pipeline_type,
        creator_id=get_current_user_id(),
        request_payload=data,
    )

    # If runNow, submit to inference queue
    if data.get("runNow"):
        from flask import current_app
        from app.services.inference_queue import inference_queue
        inference_queue.submit(current_app._get_current_object(), task.id)

    return success({
        "id": task.id,
        "taskName": task.task_name,
        "imageId": task.image_id,
        "pipelineType": task.pipeline_type,
        "status": task.status,
        "createTime": task.create_time.strftime("%Y-%m-%d %H:%M:%S") if task.create_time else None,
    }, msg="推理任务已创建")


@detection_bp.route("/task/list", methods=["GET"])
@login_required
def list_tasks():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)
    status = request.args.get("status", "")

    query = InferenceTask.query
    if status:
        query = query.filter_by(status=status)
    query = query.order_by(InferenceTask.create_time.desc())

    pagination = query.paginate(page=page, per_page=page_size, error_out=False)

    rows = []
    for t in pagination.items:
        rows.append({
            "id": t.id,
            "taskName": t.task_name,
            "imageId": t.image_id,
            "pipelineType": t.pipeline_type,
            "status": t.status,
            "progress": 100 if t.status == "completed" else 0,
            "totalImages": 1,
            "processedImages": 1 if t.status == "completed" else 0,
            "createTime": t.create_time.strftime("%Y-%m-%d %H:%M:%S") if t.create_time else None,
            "finishTime": t.finish_time.strftime("%Y-%m-%d %H:%M:%S") if t.finish_time else None,
        })

    return table_data(rows=rows, total=pagination.total)


@detection_bp.route("/task/<int:task_id>", methods=["GET"])
@login_required
def get_task(task_id):
    task = InferenceTask.query.get_or_404(task_id)
    return success({
        "id": task.id,
        "taskName": task.task_name,
        "imageId": task.image_id,
        "pipelineType": task.pipeline_type,
        "status": task.status,
        "progress": 100 if task.status == "completed" else 0,
        "totalImages": 1,
        "processedImages": 1 if task.status == "completed" else 0,
        "createTime": task.create_time.strftime("%Y-%m-%d %H:%M:%S") if task.create_time else None,
        "finishTime": task.finish_time.strftime("%Y-%m-%d %H:%M:%S") if task.finish_time else None,
    })


@detection_bp.route("/task/<int:task_id>", methods=["DELETE"])
@login_required
def delete_task(task_id):
    task = InferenceTask.query.get_or_404(task_id)
    if task.status == "running":
        return error("运行中的任务无法删除", 400)
    db.session.delete(task)
    db.session.commit()
    return success(msg="删除成功")


# ── Result endpoints (now backed by HazardReasoningResult) ────────────────

@detection_bp.route("/result/<int:task_id>", methods=["GET"])
@login_required
def list_results(task_id):
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 20, type=int)

    query = HazardReasoningResult.query.filter_by(task_id=task_id).order_by(HazardReasoningResult.id.desc())
    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data(
        rows=[_hazard_to_result_row(r) for r in pagination.items],
        total=pagination.total,
    )


@detection_bp.route("/result/<int:task_id>/<int:image_id>", methods=["GET"])
@login_required
def get_result_detail(task_id, image_id):
    """Get detection results for a specific image, with image info."""
    hazards = HazardReasoningResult.query.filter_by(task_id=task_id, image_id=image_id).all()
    image = ImageAsset.query.get(image_id)

    # Build detection list with associated DetectedObject for bbox
    detections = []
    for h in hazards:
        obj = DetectedObject.query.get(h.worker_object_id) if h.worker_object_id else None
        detections.append(_hazard_to_detail_detection(h, obj))

    # Count unsafe
    unsafe_count = sum(
        1 for d in detections
        if d["primaryBehavior"] not in ("safe", "unknown") and d["confidence"] > 0.3
    )

    return success({
        "image": image.to_dict() if image else None,
        "detections": detections,
        "totalWorkers": len(detections),
        "unsafeCount": unsafe_count,
    })


# ── Single image quick detection (kept from original) ─────────────────────

@detection_bp.route("/single", methods=["POST"])
@login_required
def single_detection():
    """Quick single-image detection (synchronous, for demo)."""
    if "file" not in request.files:
        return error("请上传图片", 400)

    from app.utils.file_utils import save_uploaded_file
    from app.services.detection_service import detect_single_image

    file = request.files["file"]
    rel_path = save_uploaded_file(file)
    if not rel_path:
        return error("不支持的图片格式", 400)

    try:
        results = detect_single_image(file)
    except Exception as e:
        return error(f"检测失败: {str(e)}", 500)

    return success(results, msg="检测完成")


# ── WebSocket for task progress ───────────────────────────────────────────

@socketio.on("subscribe_task")
def handle_subscribe(data):
    task_id = data.get("taskId")
    if task_id:
        from flask_socketio import join_room
        join_room(f"task_{task_id}")
        task = InferenceTask.query.get(task_id)
        if task:
            emit("task_progress", {
                "id": task.id,
                "status": task.status,
                "currentStage": task.current_stage,
                "createTime": task.create_time.strftime("%Y-%m-%d %H:%M:%S") if task.create_time else None,
                "finishTime": task.finish_time.strftime("%Y-%m-%d %H:%M:%S") if task.finish_time else None,
            }, to=f"task_{task_id}")
