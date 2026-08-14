"""
Async tasks for detection and report generation.
"""

import json
import time
from datetime import datetime

from app.extensions import db, socketio
from app.models.detection import DetectionTask, DetectionResult
from app.models.dataset import DatasetImage
from app.models.report import EvaluationReport
from app.services.detection_service import detect_dataset_images, detect_single_image
from app.services.report_service import generate_evaluation_report, export_report_excel


def run_detection_task_sync(task_id):
    """Run detection on all images in a dataset (synchronous, no Celery needed)."""
    task = DetectionTask.query.get(task_id)
    if not task:
        return {"error": "Task not found"}

    task.status = "running"
    db.session.commit()

    try:
        processed = 0
        for image_id, results in detect_dataset_images(task.dataset_id, task.model_id):
            for det in results["detections"]:
                result = DetectionResult(
                    task_id=task.id,
                    image_id=image_id,
                    worker_bbox=det["workerBbox"],
                    behavior_probs=json.dumps(det["behaviorProbs"]),
                    risk_probs=json.dumps(det["riskProbs"]),
                    risk_severity=det["riskSeverity"],
                    primary_behavior=det["primaryBehavior"],
                    primary_risk=det["primaryRisk"],
                    confidence=det["confidence"],
                )
                db.session.add(result)

            processed += 1
            task.processed_images = processed
            task.progress = round(processed / task.total_images * 100, 2) if task.total_images else 0
            db.session.commit()

            time.sleep(0.05)

        task.status = "completed"
        task.finish_time = datetime.now()
        db.session.commit()

    except Exception as e:
        task.status = "failed"
        db.session.commit()
        raise

    return {"task_id": task_id, "status": "completed"}
