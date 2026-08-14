"""
Report service: generate evaluation reports from inference pipeline data.
"""

import json
import os
from datetime import datetime


def generate_evaluation_report(task_id, model_id=None):
    """
    Generate a comprehensive evaluation report from inference pipeline results.

    Returns a dict with all report sections.
    """
    from app.models.inference import (
        HazardImageSummary,
        HazardReasoningResult,
        InferenceTask,
    )
    from app import create_app

    app = create_app()
    with app.app_context():
        task = InferenceTask.query.get(task_id)
        if not task:
            return {"status": "failed", "error": f"任务 #{task_id} 不存在"}

        worker_results = HazardReasoningResult.query.filter_by(task_id=task_id).all()
        summary = HazardImageSummary.query.filter_by(task_id=task_id).order_by(
            HazardImageSummary.id.desc()
        ).first()

        # Per-category stats
        behavior_stats = {}
        for r in worker_results:
            cat = r.unsafe_behavior_category or "unknown"
            if cat not in behavior_stats:
                behavior_stats[cat] = {"count": 0, "total_score": 0, "high_confidence": 0}
            behavior_stats[cat]["count"] += 1
            behavior_stats[cat]["total_score"] += r.pred_score
            if r.pred_score > 0.7:
                behavior_stats[cat]["high_confidence"] += 1

        # Unsafe workers
        unsafe_results = [
            r for r in worker_results
            if r.pred_score > 0.3 and r.unsafe_behavior_category not in ("safe", "unknown", None)
        ]
        avg_severity = (
            sum(r.pred_score for r in unsafe_results) / len(unsafe_results)
            if unsafe_results else 0
        )

        report = {
            "status": "completed",
            "meta": {
                "taskId": task_id,
                "taskName": task.task_name,
                "pipelineType": task.pipeline_type,
                "generatedAt": datetime.now().isoformat(),
                "totalResults": len(worker_results),
                "unsafeDetections": len(unsafe_results),
            },
            "summary": {
                "totalWorkersDetected": len(worker_results),
                "unsafeBehaviorCount": len(unsafe_results),
                "unsafeRate": round(len(unsafe_results) / len(worker_results), 4) if worker_results else 0,
                "averageRiskSeverity": round(avg_severity, 4),
                "hasUnsafeBehavior": summary.has_unsafe_behavior if summary else False,
                "highestRiskCategory": summary.highest_risk_category if summary else None,
            },
            "perClassStats": {
                b: {
                    "count": s["count"],
                    "avgConfidence": round(s["total_score"] / s["count"], 4) if s["count"] > 0 else 0,
                    "highConfidenceRate": round(s["high_confidence"] / s["count"], 4) if s["count"] > 0 else 0,
                }
                for b, s in behavior_stats.items()
            },
            "imageLevelResult": summary.to_dict() if summary else None,
            "workerResults": [
                {
                    "id": r.id,
                    "imageId": r.image_id,
                    "category": r.unsafe_behavior_category,
                    "predLabel": r.pred_major_label,
                    "predScore": r.pred_score,
                    "contextText": r.context_text,
                }
                for r in worker_results[:100]
            ],
        }

        return report


def export_report_excel(report_data, output_path):
    """Export report as Excel file."""
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "检测报告"

        ws.append(["检测报告"])
        ws.append(["生成时间", report_data["meta"]["generatedAt"]])
        ws.append(["任务", report_data["meta"]["taskName"]])
        ws.append(["流水线类型", report_data["meta"]["pipelineType"]])
        ws.append(["总检测数", report_data["summary"]["totalWorkersDetected"]])
        ws.append(["不安全行为数", report_data["summary"]["unsafeBehaviorCount"]])
        ws.append(["不安全率", report_data["summary"]["unsafeRate"]])
        ws.append(["平均风险等级", report_data["summary"]["averageRiskSeverity"]])
        ws.append([])
        ws.append(["行为类别", "检测数", "平均置信度", "高置信率"])

        for behavior, stats in report_data.get("perClassStats", {}).items():
            ws.append([behavior, stats["count"], stats["avgConfidence"], stats["highConfidenceRate"]])

        wb.save(output_path)
        return output_path
    except ImportError:
        return None
