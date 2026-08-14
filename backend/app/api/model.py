import json
import os

from flask import Blueprint, current_app, request

from app.extensions import db
from app.models.model_registry import ModelRegistry, TrainingTask
from app.utils.response import success, error, table_data
from app.utils.auth import login_required
from app.utils.file_utils import save_uploaded_file

model_bp = Blueprint("model", __name__)


@model_bp.route("/list", methods=["GET"])
@login_required
def list_models():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)

    query = ModelRegistry.query.order_by(ModelRegistry.create_time.desc())
    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data(
        rows=[m.to_dict() for m in pagination.items],
        total=pagination.total,
    )


@model_bp.route("/register", methods=["POST"])
@login_required
def register_model():
    data = request.get_json() or {}
    model = ModelRegistry(
        name=data["name"],
        version=data.get("version", "1.0.0"),
        checkpoint_path=data["checkpointPath"],
        config_json=json.dumps(data.get("configJson", {}), ensure_ascii=False),
        metrics_json=json.dumps(data.get("metricsJson", {}), ensure_ascii=False),
        service_type=data.get("serviceType", "yolo_detection"),
        runtime_type=data.get("runtimeType", "local_python"),
    )
    db.session.add(model)
    db.session.commit()
    return success(model.to_dict(), msg="模型注册成功")


@model_bp.route("/<int:model_id>", methods=["GET"])
@login_required
def get_model(model_id):
    model = ModelRegistry.query.get_or_404(model_id)
    return success(model.to_dict())


@model_bp.route("/<int:model_id>/activate", methods=["POST"])
@login_required
def activate_model(model_id):
    model = ModelRegistry.query.get_or_404(model_id)
    ModelRegistry.query.filter(ModelRegistry.id != model_id).update({"status": "archived"})
    model.status = "active"
    db.session.commit()
    return success(model.to_dict(), msg=f"模型 {model.name} 已激活")


@model_bp.route("/<int:model_id>/detect", methods=["POST"])
@login_required
def detect_with_registered_model(model_id):
    """Run YOLO detection using a registered model on an uploaded image."""
    if "file" not in request.files:
        return error("请上传图片", 400)

    file = request.files["file"]
    rel_path = save_uploaded_file(file, subfolder="detections")
    if not rel_path:
        return error("不支持的图片格式", 400)

    full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], rel_path)

    try:
        from app.services.detection_service import detect_with_model
        detections = detect_with_model(model_id, full_path)
    except ValueError as e:
        return error(str(e), 400)
    except FileNotFoundError as e:
        return error(str(e), 404)
    except Exception as e:
        return error(f"检测失败: {str(e)}", 500)

    return success({
        "modelId": model_id,
        "imageUrl": f"/uploads/{rel_path}",
        "detections": detections,
        "totalDetections": len(detections),
    }, msg="检测完成")


@model_bp.route("/<int:model_id>", methods=["PUT"])
@login_required
def update_model(model_id):
    """Update a registered model's metadata and config."""
    model = ModelRegistry.query.get_or_404(model_id)
    data = request.get_json() or {}

    model.name = data.get("name", model.name)
    model.version = data.get("version", model.version)
    model.checkpoint_path = data.get("checkpointPath", model.checkpoint_path)
    if "configJson" in data:
        model.config_json = json.dumps(data["configJson"], ensure_ascii=False)
    if "metricsJson" in data:
        model.metrics_json = json.dumps(data["metricsJson"], ensure_ascii=False)
    if "serviceType" in data:
        model.service_type = data["serviceType"]

    db.session.commit()

    # Clear cache so next inference reloads the updated model
    try:
        from app.services.detection_service import clear_model_cache
        clear_model_cache(model_id)
    except Exception:
        pass

    return success(model.to_dict(), msg="模型更新成功")


@model_bp.route("/<int:model_id>", methods=["DELETE"])
@model_bp.route("/<int:model_id>/delete", methods=["DELETE"])
@login_required
def delete_model(model_id):
    model = ModelRegistry.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()

    # Clear model cache
    try:
        from app.services.detection_service import clear_model_cache
        clear_model_cache(model_id)
    except Exception:
        pass

    return success(msg="删除成功")


@model_bp.route("/training", methods=["POST"])
@login_required
def create_training():
    data = request.get_json() or {}
    task = TrainingTask(
        model_id=data.get("modelId"),
        dataset_id=data["datasetId"],
        config_json=json.dumps(data.get("configJson", {}), ensure_ascii=False),
    )
    db.session.add(task)
    db.session.commit()

    from app.tasks.training_tasks import run_training_task
    run_training_task.delay(task.id)

    return success(task.to_dict(), msg="训练任务已创建")


@model_bp.route("/training/list", methods=["GET"])
@login_required
def list_trainings():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)

    query = TrainingTask.query.order_by(TrainingTask.start_time.desc())
    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data(
        rows=[t.to_dict() for t in pagination.items],
        total=pagination.total,
    )


@model_bp.route("/training/<int:task_id>", methods=["GET"])
@login_required
def get_training(task_id):
    task = TrainingTask.query.get_or_404(task_id)
    return success(task.to_dict())

