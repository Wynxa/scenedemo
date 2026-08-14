import hashlib
import json
from datetime import datetime

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
from app.services.algorithm_adapters import get_algorithm_adapters


STAGE_SERVICE_DEFAULTS = {
    "detector": "detector_service",
    "scene_graph": "scene_graph_service",
    "reasoning": "hazard_reasoning_service",
}


class InferenceOrchestrator:
    @staticmethod
    def compute_file_md5(file_path: str) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def create_inference_task(*, image_id: int, pipeline_type: str, creator_id: int, request_payload: dict):
        task_name = request_payload.get("taskName") or f"{pipeline_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = InferenceTask(
            task_name=task_name,
            image_id=image_id,
            pipeline_type=pipeline_type,
            status="pending",
            current_stage="detector" if pipeline_type == "full_pipeline" else pipeline_type,
            request_json=json.dumps(request_payload, ensure_ascii=False),
            creator_id=creator_id,
        )
        db.session.add(task)
        db.session.flush()

        stage_specs = []
        if pipeline_type == "full_pipeline":
            stage_specs = [
                ("detector", request_payload.get("detectorModelId")),
                ("scene_graph", request_payload.get("sceneGraphModelId")),
                ("reasoning", request_payload.get("reasoningModelId")),
            ]
        elif pipeline_type == "scene_graph_only":
            stage_specs = [("scene_graph", request_payload.get("sceneGraphModelId"))]
        elif pipeline_type == "reasoning_only":
            stage_specs = [("reasoning", request_payload.get("reasoningModelId"))]
        elif pipeline_type == "detector_only":
            stage_specs = [("detector", request_payload.get("detectorModelId"))]

        stage_runs = []
        for stage_name, model_id in stage_specs:
            stage_run = InferenceStageRun(
                task_id=task.id,
                stage_name=stage_name,
                service_name=STAGE_SERVICE_DEFAULTS.get(stage_name),
                model_id=model_id,
                status="pending",
                input_json=json.dumps(request_payload.get("config", {}).get(stage_name, {}), ensure_ascii=False),
            )
            db.session.add(stage_run)
            stage_runs.append(stage_run)

        db.session.commit()
        return task, stage_runs

    @staticmethod
    def update_task_status(task: InferenceTask, *, status: str, current_stage: str | None = None, error_message: str | None = None):
        task.status = status
        if current_stage is not None:
            task.current_stage = current_stage
        if error_message is not None:
            task.error_message = error_message
        if status in {"completed", "failed"}:
            task.finish_time = datetime.now()
        db.session.commit()
        return task

    @staticmethod
    def update_stage_run(stage_run: InferenceStageRun, *, status: str, output_payload: dict | None = None, error_message: str | None = None):
        stage_run.status = status
        if stage_run.start_time is None:
            stage_run.start_time = datetime.now()
        if output_payload is not None:
            stage_run.output_json = json.dumps(output_payload, ensure_ascii=False)
        if error_message is not None:
            stage_run.error_message = error_message
        if status in {"completed", "failed"}:
            stage_run.finish_time = datetime.now()
        db.session.commit()
        return stage_run

    @staticmethod
    def save_detector_objects(*, task_id: int, stage_run_id: int, image_id: int, objects: list[dict]):
        rows = []
        for index, item in enumerate(objects):
            bbox = item.get("bbox") or item.get("bbox_xyxy") or [0, 0, 0, 0]
            row = DetectedObject(
                task_id=task_id,
                stage_run_id=stage_run_id,
                image_id=image_id,
                object_index=index,
                label_id=item.get("label_id"),
                label_name=item.get("label") or item.get("name"),
                bbox_x1=float(bbox[0]),
                bbox_y1=float(bbox[1]),
                bbox_x2=float(bbox[2]),
                bbox_y2=float(bbox[3]),
                score=float(item.get("score", 0.0)),
                source_model=item.get("model_name"),
                source_branch=item.get("source"),
                extra_json=json.dumps(item, ensure_ascii=False),
            )
            db.session.add(row)
            rows.append(row)
        db.session.commit()
        return rows

    @staticmethod
    def _resolve_detected_object_ids(task_id: int, object_id_map: dict[int, int] | None = None) -> dict[int, int]:
        """Translate scene-graph object indexes into detected_object primary keys."""
        if object_id_map is not None:
            return object_id_map
        rows = DetectedObject.query.filter_by(task_id=task_id).all()
        return {int(row.object_index): int(row.id) for row in rows}

    @staticmethod
    def save_scene_graph_relations(
        *, task_id: int, stage_run_id: int, image_id: int, relationships: list[dict], object_id_map: dict[int, int] | None = None
    ):
        object_id_map = InferenceOrchestrator._resolve_detected_object_ids(task_id, object_id_map)
        rows = []
        for rel in relationships:
            subject_id = object_id_map.get(int(rel["subject_id"])) if rel.get("subject_id") is not None else None
            object_id = object_id_map.get(int(rel["object_id"])) if rel.get("object_id") is not None else None
            row = SceneGraphRelation(
                task_id=task_id,
                stage_run_id=stage_run_id,
                image_id=image_id,
                subject_object_id=subject_id,
                object_object_id=object_id,
                predicate_id=rel.get("predicate_id"),
                predicate_name=rel.get("predicate"),
                score=float(rel.get("score", 0.0)),
                all_scores_json=json.dumps(rel.get("all_scores", []), ensure_ascii=False),
            )
            db.session.add(row)
            rows.append(row)
        db.session.commit()
        return rows

    @staticmethod
    def save_hazard_results(
        *, task_id: int, stage_run_id: int, image_id: int, worker_results: list[dict], image_level_result: dict,
        object_id_map: dict[int, int] | None = None,
    ):
        object_id_map = InferenceOrchestrator._resolve_detected_object_ids(task_id, object_id_map)
        rows = []
        for item in worker_results:
            row = HazardReasoningResult(
                task_id=task_id,
                stage_run_id=stage_run_id,
                image_id=image_id,
                worker_object_id=object_id_map.get(int(item["worker_id"])) if item.get("worker_id") is not None else None,
                context_id=item.get("context_id"),
                context_text=item.get("context_text"),
                relation_type=item.get("relation_type"),
                unsafe_behavior_category=item.get("unsafe_behavior_category"),
                pred_major_label=item.get("pred_major_label"),
                pred_major_label_id=item.get("pred_major_label_id"),
                pred_score=float(item.get("pred_score", 0.0)),
                required_ppe_json=json.dumps(item.get("required_ppe", []), ensure_ascii=False),
                present_ppe_json=json.dumps(item.get("present_ppe", []), ensure_ascii=False),
                missing_ppe_json=json.dumps(item.get("missing_ppe", []), ensure_ascii=False),
                scene_text=item.get("scene_text"),
                scene_text_structured=item.get("scene_text_structured"),
                topk_json=json.dumps(item.get("topk", []), ensure_ascii=False),
            )
            db.session.add(row)
            rows.append(row)

        summary = HazardImageSummary(
            task_id=task_id,
            image_id=image_id,
            has_unsafe_behavior=bool(image_level_result.get("has_unsafe_behavior", False)),
            unsafe_worker_count=int(image_level_result.get("unsafe_worker_count", 0)),
            highest_risk_category=(image_level_result.get("highest_risk_prediction") or {}).get("unsafe_behavior_category"),
            highest_risk_score=float(((image_level_result.get("highest_risk_prediction") or {}).get("pred_score") or 0.0)),
            unsafe_behavior_results_json=json.dumps(image_level_result.get("unsafe_behavior_categories", []), ensure_ascii=False),
        )
        db.session.add(summary)
        db.session.commit()
        return rows, summary

    @staticmethod
    def execute_full_pipeline(task: InferenceTask) -> dict:
        if task.image is None:
            raise ValueError("任务未关联图片")

        stage_runs = {row.stage_name: row for row in task.stage_runs.all()}
        adapters = get_algorithm_adapters()
        image_path = task.image.file_path
        from flask import current_app
        full_image_path = image_path
        if not image_path.startswith(("/", "\\")) and ":" not in image_path:
            full_image_path = str((current_app.config["UPLOAD_FOLDER"] + "/" + image_path).replace("//", "/"))

        InferenceOrchestrator.update_task_status(task, status="running", current_stage="detector")

        detector_payload = adapters.run_detector(image_path=full_image_path, image_id=str(task.image.id))
        detector_stage = stage_runs["detector"]
        DetectedObject.query.filter_by(task_id=task.id, stage_run_id=detector_stage.id).delete()
        detected_rows = InferenceOrchestrator.save_detector_objects(
            task_id=task.id, stage_run_id=detector_stage.id, image_id=task.image_id, objects=detector_payload.get("objects", [])
        )
        object_id_map = {
            int(item.get("object_id", index)): int(row.id)
            for index, (item, row) in enumerate(zip(detector_payload.get("objects", []), detected_rows))
        }
        InferenceOrchestrator.update_stage_run(detector_stage, status="completed", output_payload=detector_payload)

        InferenceOrchestrator.update_task_status(task, status="running", current_stage="scene_graph")
        scene_graph_payload = adapters.run_scene_graph(
            image_path=detector_payload["image_path"],
            image_id=detector_payload["image_id"],
            objects=detector_payload.get("objects", []),
        )
        scene_graph_stage = stage_runs["scene_graph"]
        SceneGraphRelation.query.filter_by(task_id=task.id, stage_run_id=scene_graph_stage.id).delete()
        InferenceOrchestrator.save_scene_graph_relations(
            task_id=task.id, stage_run_id=scene_graph_stage.id, image_id=task.image_id,
            relationships=scene_graph_payload.get("relationships", []), object_id_map=object_id_map,
        )
        InferenceOrchestrator.update_stage_run(scene_graph_stage, status="completed", output_payload=scene_graph_payload)

        InferenceOrchestrator.update_task_status(task, status="running", current_stage="reasoning")
        hazard_payload = adapters.run_hazard_reasoning(scene_graph_payload)
        reasoning_stage = stage_runs["reasoning"]
        HazardReasoningResult.query.filter_by(task_id=task.id, stage_run_id=reasoning_stage.id).delete()
        HazardImageSummary.query.filter_by(task_id=task.id, image_id=task.image_id).delete()
        InferenceOrchestrator.save_hazard_results(
            task_id=task.id,
            stage_run_id=reasoning_stage.id,
            image_id=task.image_id,
            worker_results=hazard_payload.get("worker_results", []),
            image_level_result=hazard_payload.get("image_level_result", {}),
            object_id_map=object_id_map,
        )
        InferenceOrchestrator.update_stage_run(reasoning_stage, status="completed", output_payload=hazard_payload)

        summary_payload = {
            "unsafeBehaviorResults": hazard_payload.get("unsafe_behavior_results", []),
            "imageLevelResult": hazard_payload.get("image_level_result", {}),
        }
        task.result_summary_json = json.dumps(summary_payload, ensure_ascii=False)
        InferenceOrchestrator.update_task_status(task, status="completed", current_stage="done")
        return {
            "detector": detector_payload,
            "sceneGraph": scene_graph_payload,
            "reasoning": hazard_payload,
        }
