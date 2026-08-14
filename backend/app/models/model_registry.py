from datetime import datetime
from app.extensions import db


class ModelRegistry(db.Model):
    __tablename__ = "model_registry"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), nullable=False)
    version = db.Column(db.String(32), default="1.0.0")
    checkpoint_path = db.Column(db.String(512), nullable=False)
    config_json = db.Column(db.Text)  # ModelConfig JSON
    service_type = db.Column(db.String(32), default="generic")
    runtime_type = db.Column(db.String(32), default="local_python")
    metrics_json = db.Column(db.Text)  # Evaluation metrics JSON
    status = db.Column(db.String(16), default="active")  # active / archived
    create_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        import json

        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "checkpointPath": self.checkpoint_path,
            "configJson": json.loads(self.config_json) if self.config_json else {},
            "serviceType": self.service_type,
            "runtimeType": self.runtime_type,
            "metricsJson": json.loads(self.metrics_json) if self.metrics_json else {},
            "status": self.status,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }


class TrainingTask(db.Model):
    __tablename__ = "training_task"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_id = db.Column(db.Integer)  # target model registry ID
    dataset_id = db.Column(db.Integer, nullable=False)
    config_json = db.Column(db.Text)  # Training hyperparameters JSON
    status = db.Column(db.String(16), default="pending")  # pending / running / completed / failed
    metrics_json = db.Column(db.Text)  # Result metrics JSON
    log_path = db.Column(db.String(512))
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)

    def to_dict(self):
        import json

        return {
            "id": self.id,
            "modelId": self.model_id,
            "datasetId": self.dataset_id,
            "configJson": json.loads(self.config_json) if self.config_json else {},
            "status": self.status,
            "metricsJson": json.loads(self.metrics_json) if self.metrics_json else {},
            "logPath": self.log_path,
            "startTime": self.start_time.strftime("%Y-%m-%d %H:%M:%S") if self.start_time else None,
            "endTime": self.end_time.strftime("%Y-%m-%d %H:%M:%S") if self.end_time else None,
        }
