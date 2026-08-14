from datetime import datetime
import json

from app.extensions import db


class ImageAsset(db.Model):
    __tablename__ = "image_asset"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.Integer, nullable=True, index=True)
    source_type = db.Column(db.String(32), default="upload")
    file_name = db.Column(db.String(256), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_url = db.Column(db.String(512))
    width = db.Column(db.Integer, default=0)
    height = db.Column(db.Integer, default=0)
    file_size = db.Column(db.BigInteger, default=0)
    md5 = db.Column(db.String(64))
    status = db.Column(db.String(16), default="active")
    creator_id = db.Column(db.Integer, default=0)
    create_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "datasetId": self.dataset_id,
            "sourceType": self.source_type,
            "fileName": self.file_name,
            "filePath": self.file_path,
            "fileUrl": self.file_url,
            "width": self.width,
            "height": self.height,
            "fileSize": self.file_size,
            "md5": self.md5,
            "status": self.status,
            "creatorId": self.creator_id,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }


class InferenceTask(db.Model):
    __tablename__ = "inference_task"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_name = db.Column(db.String(128))
    image_id = db.Column(db.Integer, db.ForeignKey("image_asset.id", ondelete="SET NULL"), nullable=True, index=True)
    pipeline_type = db.Column(db.String(32), default="full_pipeline")
    status = db.Column(db.String(16), default="pending")
    current_stage = db.Column(db.String(32), default="detector")
    request_json = db.Column(db.Text)
    result_summary_json = db.Column(db.Text)
    error_message = db.Column(db.Text)
    creator_id = db.Column(db.Integer, default=0)
    create_time = db.Column(db.DateTime, default=datetime.now)
    finish_time = db.Column(db.DateTime)

    image = db.relationship("ImageAsset", backref=db.backref("inference_tasks", lazy="dynamic"))
    stage_runs = db.relationship("InferenceStageRun", backref="task", lazy="dynamic", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "taskName": self.task_name,
            "imageId": self.image_id,
            "pipelineType": self.pipeline_type,
            "status": self.status,
            "currentStage": self.current_stage,
            "requestJson": json.loads(self.request_json) if self.request_json else {},
            "resultSummaryJson": json.loads(self.result_summary_json) if self.result_summary_json else {},
            "errorMessage": self.error_message,
            "creatorId": self.creator_id,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
            "finishTime": self.finish_time.strftime("%Y-%m-%d %H:%M:%S") if self.finish_time else None,
        }


class InferenceStageRun(db.Model):
    __tablename__ = "inference_stage_run"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey("inference_task.id", ondelete="CASCADE"), nullable=False, index=True)
    stage_name = db.Column(db.String(32), nullable=False)
    service_name = db.Column(db.String(64))
    model_id = db.Column(db.Integer, nullable=True, index=True)
    status = db.Column(db.String(16), default="pending")
    input_json = db.Column(db.Text)
    output_json = db.Column(db.Text)
    metrics_json = db.Column(db.Text)
    error_message = db.Column(db.Text)
    start_time = db.Column(db.DateTime)
    finish_time = db.Column(db.DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "taskId": self.task_id,
            "stageName": self.stage_name,
            "serviceName": self.service_name,
            "modelId": self.model_id,
            "status": self.status,
            "inputJson": json.loads(self.input_json) if self.input_json else {},
            "outputJson": json.loads(self.output_json) if self.output_json else {},
            "metricsJson": json.loads(self.metrics_json) if self.metrics_json else {},
            "errorMessage": self.error_message,
            "startTime": self.start_time.strftime("%Y-%m-%d %H:%M:%S") if self.start_time else None,
            "finishTime": self.finish_time.strftime("%Y-%m-%d %H:%M:%S") if self.finish_time else None,
        }


class DetectedObject(db.Model):
    __tablename__ = "detected_object"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey("inference_task.id", ondelete="CASCADE"), nullable=False, index=True)
    stage_run_id = db.Column(db.Integer, db.ForeignKey("inference_stage_run.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image_asset.id", ondelete="SET NULL"), nullable=True, index=True)
    object_index = db.Column(db.Integer, default=0)
    label_id = db.Column(db.Integer)
    label_name = db.Column(db.String(64), nullable=False)
    bbox_x1 = db.Column(db.Float, default=0.0)
    bbox_y1 = db.Column(db.Float, default=0.0)
    bbox_x2 = db.Column(db.Float, default=0.0)
    bbox_y2 = db.Column(db.Float, default=0.0)
    score = db.Column(db.Float, default=0.0)
    source_model = db.Column(db.String(128))
    source_branch = db.Column(db.String(64))
    extra_json = db.Column(db.Text)
    create_time = db.Column(db.DateTime, default=datetime.now)

    stage_run = db.relationship("InferenceStageRun", backref=db.backref("detected_objects", lazy="dynamic", cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            "id": self.id,
            "taskId": self.task_id,
            "stageRunId": self.stage_run_id,
            "imageId": self.image_id,
            "objectIndex": self.object_index,
            "labelId": self.label_id,
            "labelName": self.label_name,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "score": self.score,
            "sourceModel": self.source_model,
            "sourceBranch": self.source_branch,
            "extraJson": json.loads(self.extra_json) if self.extra_json else {},
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }


class SceneGraphRelation(db.Model):
    __tablename__ = "scene_graph_relation"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey("inference_task.id", ondelete="CASCADE"), nullable=False, index=True)
    stage_run_id = db.Column(db.Integer, db.ForeignKey("inference_stage_run.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image_asset.id", ondelete="SET NULL"), nullable=True, index=True)
    subject_object_id = db.Column(db.Integer, db.ForeignKey("detected_object.id", ondelete="SET NULL"), nullable=True)
    object_object_id = db.Column(db.Integer, db.ForeignKey("detected_object.id", ondelete="SET NULL"), nullable=True)
    predicate_id = db.Column(db.Integer)
    predicate_name = db.Column(db.String(64), nullable=False)
    score = db.Column(db.Float, default=0.0)
    all_scores_json = db.Column(db.Text)
    create_time = db.Column(db.DateTime, default=datetime.now)

    stage_run = db.relationship("InferenceStageRun", backref=db.backref("scene_graph_relations", lazy="dynamic", cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            "id": self.id,
            "taskId": self.task_id,
            "stageRunId": self.stage_run_id,
            "imageId": self.image_id,
            "subjectObjectId": self.subject_object_id,
            "objectObjectId": self.object_object_id,
            "predicateId": self.predicate_id,
            "predicateName": self.predicate_name,
            "score": self.score,
            "allScoresJson": json.loads(self.all_scores_json) if self.all_scores_json else [],
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }


class HazardReasoningResult(db.Model):
    __tablename__ = "hazard_reasoning_result"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey("inference_task.id", ondelete="CASCADE"), nullable=False, index=True)
    stage_run_id = db.Column(db.Integer, db.ForeignKey("inference_stage_run.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image_asset.id", ondelete="SET NULL"), nullable=True, index=True)
    worker_object_id = db.Column(db.Integer, db.ForeignKey("detected_object.id", ondelete="SET NULL"), nullable=True)
    context_id = db.Column(db.String(64))
    context_text = db.Column(db.String(128))
    relation_type = db.Column(db.String(64))
    unsafe_behavior_category = db.Column(db.String(128))
    pred_major_label = db.Column(db.String(128))
    pred_major_label_id = db.Column(db.Integer)
    pred_score = db.Column(db.Float, default=0.0)
    required_ppe_json = db.Column(db.Text)
    present_ppe_json = db.Column(db.Text)
    missing_ppe_json = db.Column(db.Text)
    scene_text = db.Column(db.Text)
    scene_text_structured = db.Column(db.Text)
    topk_json = db.Column(db.Text)
    create_time = db.Column(db.DateTime, default=datetime.now)

    stage_run = db.relationship("InferenceStageRun", backref=db.backref("hazard_results", lazy="dynamic", cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            "id": self.id,
            "taskId": self.task_id,
            "stageRunId": self.stage_run_id,
            "imageId": self.image_id,
            "workerObjectId": self.worker_object_id,
            "contextId": self.context_id,
            "contextText": self.context_text,
            "relationType": self.relation_type,
            "unsafeBehaviorCategory": self.unsafe_behavior_category,
            "predMajorLabel": self.pred_major_label,
            "predMajorLabelId": self.pred_major_label_id,
            "predScore": self.pred_score,
            "requiredPpeJson": json.loads(self.required_ppe_json) if self.required_ppe_json else [],
            "presentPpeJson": json.loads(self.present_ppe_json) if self.present_ppe_json else [],
            "missingPpeJson": json.loads(self.missing_ppe_json) if self.missing_ppe_json else [],
            "sceneText": self.scene_text,
            "sceneTextStructured": self.scene_text_structured,
            "topkJson": json.loads(self.topk_json) if self.topk_json else [],
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }


class HazardImageSummary(db.Model):
    __tablename__ = "hazard_image_summary"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.Integer, db.ForeignKey("inference_task.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image_asset.id", ondelete="SET NULL"), nullable=True, index=True)
    has_unsafe_behavior = db.Column(db.Boolean, default=False)
    unsafe_worker_count = db.Column(db.Integer, default=0)
    highest_risk_category = db.Column(db.String(128))
    highest_risk_score = db.Column(db.Float, default=0.0)
    unsafe_behavior_results_json = db.Column(db.Text)
    create_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "taskId": self.task_id,
            "imageId": self.image_id,
            "hasUnsafeBehavior": self.has_unsafe_behavior,
            "unsafeWorkerCount": self.unsafe_worker_count,
            "highestRiskCategory": self.highest_risk_category,
            "highestRiskScore": self.highest_risk_score,
            "unsafeBehaviorResultsJson": json.loads(self.unsafe_behavior_results_json) if self.unsafe_behavior_results_json else [],
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }
