"""
Celery async task for model training.

In production, this would call train.py with the appropriate config.
For the prototype, this is a placeholder that simulates training.
"""

import json
import time
from datetime import datetime

from app.extensions import celery, db, socketio
from app.models.model_registry import TrainingTask, ModelRegistry


@celery.task(bind=True, max_retries=1)
def run_training_task(self, training_id):
    """Simulate model training (prototype)."""
    from app import create_app

    app = create_app()
    with app.app_context():
        task = TrainingTask.query.get(training_id)
        if not task:
            return {"error": "Training task not found"}

        task.status = "running"
        task.start_time = datetime.now()
        db.session.commit()

        try:
            # Simulate training epochs
            for epoch in range(1, 21):
                time.sleep(0.5)  # Simulate epoch time

                # Simulate improving metrics
                task.metrics_json = json.dumps({
                    "epoch": epoch,
                    "train_loss": round(0.8 * (0.95 ** epoch) + 0.05, 4),
                    "val_loss": round(0.7 * (0.93 ** epoch) + 0.1, 4),
                    "behavior_f1": round(min(0.85, 0.3 + 0.03 * epoch), 4),
                    "risk_f1": round(min(0.80, 0.25 + 0.035 * epoch), 4),
                    "severity_mae": round(max(0.05, 0.3 - 0.012 * epoch), 4),
                })
                db.session.commit()

            task.status = "completed"
            task.end_time = datetime.now()

            # Auto-register model
            config = json.loads(task.config_json) if task.config_json else {}
            model = ModelRegistry(
                name=f"Training-Result-{training_id}",
                version="1.0.0",
                checkpoint_path=f"checkpoints/training_{training_id}.pt",
                config_json=task.config_json,
                metrics_json=task.metrics_json,
            )
            db.session.add(model)
            db.session.commit()

        except Exception as e:
            task.status = "failed"
            task.end_time = datetime.now()
            db.session.commit()
            raise

    return {"training_id": training_id, "status": "completed"}
