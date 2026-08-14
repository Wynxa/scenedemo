import json
import os

from flask import Blueprint, request, send_file, current_app

from app.extensions import db
from app.models.inference import (
    DetectedObject,
    HazardImageSummary,
    HazardReasoningResult,
    ImageAsset,
    InferenceTask,
    SceneGraphRelation,
)
from app.models.report import EvaluationReport
from app.utils.response import success, error, table_data
from app.utils.auth import login_required, get_current_user_id

report_bp = Blueprint("report", __name__)


@report_bp.route("/generate", methods=["POST"])
@login_required
def generate_report():
    data = request.get_json()
    task_id = data.get("taskId")
    report_type = data.get("reportType", "evaluation")

    report = EvaluationReport(
        task_id=task_id,
        model_id=None,
        report_type=report_type,
        content=json.dumps({
            "status": "generating",
            "type": report_type,
        }),
        creator_id=get_current_user_id(),
    )
    db.session.add(report)
    db.session.commit()

    from app.services.report_service import generate_evaluation_report, export_report_excel
    try:
        report_data = generate_evaluation_report(report.task_id)
        report.content = json.dumps(report_data)

        output_dir = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
        excel_path = os.path.join(output_dir, "reports", f"report_{report.id}.xlsx")
        export_report_excel(report_data, excel_path)
        report.file_path = excel_path
        db.session.commit()
    except Exception as e:
        report.content = json.dumps({"status": "failed", "error": str(e)})
        db.session.commit()
        return error(f"报告生成失败: {str(e)}", 500)

    return success(report.to_dict(), msg="报告生成成功")


@report_bp.route("/list", methods=["GET"])
@login_required
def list_reports():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)

    query = EvaluationReport.query.order_by(EvaluationReport.create_time.desc())
    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data(
        rows=[r.to_dict() for r in pagination.items],
        total=pagination.total,
    )


@report_bp.route("/<int:report_id>", methods=["GET"])
@login_required
def get_report(report_id):
    report = EvaluationReport.query.get_or_404(report_id)

    # Fetch associated inference pipeline data
    task_id = report.task_id
    task = InferenceTask.query.get(task_id) if task_id else None
    image = ImageAsset.query.get(task.image_id) if task and task.image_id else None
    objects = DetectedObject.query.filter_by(task_id=task_id).order_by(DetectedObject.object_index.asc()).all() if task_id else []
    relations = SceneGraphRelation.query.filter_by(task_id=task_id).order_by(SceneGraphRelation.id.asc()).all() if task_id else []
    hazards = HazardReasoningResult.query.filter_by(task_id=task_id).order_by(HazardReasoningResult.id.asc()).all() if task_id else []
    summary = HazardImageSummary.query.filter_by(task_id=task_id).order_by(HazardImageSummary.id.desc()).first() if task_id else None

    return success({
        "report": report.to_dict(),
        "task": task.to_dict() if task else None,
        "image": image.to_dict() if image else None,
        "objects": [o.to_dict() for o in objects],
        "relations": [r.to_dict() for r in relations],
        "hazards": [h.to_dict() for h in hazards],
        "summary": summary.to_dict() if summary else None,
    })


@report_bp.route("/<int:report_id>/download", methods=["GET"])
@login_required
def download_report(report_id):
    report = EvaluationReport.query.get_or_404(report_id)
    if not report.file_path:
        return error("报告文件尚未生成", 400)
    return send_file(report.file_path, as_attachment=True)


@report_bp.route("/<int:report_id>", methods=["DELETE"])
@login_required
def delete_report(report_id):
    report = EvaluationReport.query.get_or_404(report_id)
    db.session.delete(report)
    db.session.commit()
    return success(msg="删除成功")
