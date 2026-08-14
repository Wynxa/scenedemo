from datetime import datetime
from app.extensions import db


class EvaluationReport(db.Model):
    __tablename__ = "evaluation_report"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer)
    model_id = db.Column(db.Integer)
    report_type = db.Column(db.String(32), default="evaluation")  # evaluation / comparison / ablation
    content = db.Column(db.Text)  # JSON: report data
    file_path = db.Column(db.String(512))  # exported file path (PDF/XLSX)
    creator_id = db.Column(db.Integer, default=0)
    create_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        import json

        return {
            "id": self.id,
            "taskId": self.task_id,
            "modelId": self.model_id,
            "reportType": self.report_type,
            "content": json.loads(self.content) if self.content else {},
            "filePath": self.file_path,
            "creatorId": self.creator_id,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }
