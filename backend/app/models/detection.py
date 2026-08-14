from datetime import datetime
from app.extensions import db


class DetectionTask(db.Model):
    __tablename__ = "detection_task"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.Integer, nullable=False, index=True)
    model_id = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(16), default="pending")  # pending / running / completed / failed
    config_json = db.Column(db.Text)  # JSON: detection thresholds, etc.
    progress = db.Column(db.Float, default=0.0)
    total_images = db.Column(db.Integer, default=0)
    processed_images = db.Column(db.Integer, default=0)
    creator_id = db.Column(db.Integer, default=0)
    create_time = db.Column(db.DateTime, default=datetime.now)
    finish_time = db.Column(db.DateTime)

    results = db.relationship("DetectionResult", backref="task", lazy="dynamic", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "datasetId": self.dataset_id,
            "modelId": self.model_id,
            "status": self.status,
            "configJson": self.config_json,
            "progress": self.progress,
            "totalImages": self.total_images,
            "processedImages": self.processed_images,
            "creatorId": self.creator_id,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
            "finishTime": self.finish_time.strftime("%Y-%m-%d %H:%M:%S") if self.finish_time else None,
        }


class DetectionResult(db.Model):
    __tablename__ = "detection_result"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey("detection_task.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = db.Column(db.Integer, nullable=False)
    worker_bbox = db.Column(db.String(64))  # "x,y,w,h"
    behavior_probs = db.Column(db.Text)  # JSON: {behavior_name: prob, ...}
    risk_probs = db.Column(db.Text)  # JSON: {risk_name: prob, ...}
    risk_severity = db.Column(db.Float, default=0.0)
    primary_behavior = db.Column(db.String(64))
    primary_risk = db.Column(db.String(64))
    confidence = db.Column(db.Float, default=0.0)
    scene_graph_snapshot = db.Column(db.Text)  # JSONB-like
    create_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        import json

        return {
            "id": self.id,
            "taskId": self.task_id,
            "imageId": self.image_id,
            "workerBbox": self.worker_bbox,
            "behaviorProbs": json.loads(self.behavior_probs) if self.behavior_probs else {},
            "riskProbs": json.loads(self.risk_probs) if self.risk_probs else {},
            "riskSeverity": self.risk_severity,
            "primaryBehavior": self.primary_behavior,
            "primaryRisk": self.primary_risk,
            "confidence": self.confidence,
            "sceneGraphSnapshot": json.loads(self.scene_graph_snapshot) if self.scene_graph_snapshot else None,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }
