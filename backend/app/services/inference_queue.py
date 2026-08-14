"""Bounded background queue for the CPU inference pipeline.

Long-running inference must not be executed in a Flask request handler.  The
single worker keeps model weights warm and avoids concurrent PyTorch jobs
starving the local CPU and exhausting memory.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

from app.extensions import db
from app.models.inference import InferenceTask
from app.services.inference_service import InferenceOrchestrator


class InferenceQueue:
    def __init__(self) -> None:
        self._executor = None
        self._slots = None

    def init_app(self, app) -> None:
        workers = max(1, int(app.config["INFERENCE_WORKERS"]))
        queue_size = max(workers, int(app.config["INFERENCE_QUEUE_SIZE"]))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inference")
        self._slots = BoundedSemaphore(queue_size)

    def submit(self, app, task_id: int) -> bool:
        if self._executor is None or self._slots is None:
            raise RuntimeError("Inference queue is not initialized")
        if not self._slots.acquire(blocking=False):
            return False
        self._executor.submit(self._run, app, task_id)
        return True

    def _run(self, app, task_id: int) -> None:
        try:
            with app.app_context():
                task = db.session.get(InferenceTask, task_id)
                if task is None:
                    return
                try:
                    InferenceOrchestrator.execute_full_pipeline(task)
                except Exception as exc:
                    db.session.rollback()
                    InferenceOrchestrator.update_task_status(
                        task, status="failed", current_stage=task.current_stage, error_message=str(exc)
                    )
        finally:
            self._slots.release()


inference_queue = InferenceQueue()
