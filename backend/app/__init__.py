import os
from flask import Flask, send_from_directory

from app.config import config_map
from app.extensions import init_extensions
from app.services.db_bootstrap import initialize_database
from app.services.inference_queue import inference_queue


def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__)
    app.config.from_object(config_map.get(config_name, config_map["development"]))

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["CHECKPOINT_FOLDER"], exist_ok=True)

    init_extensions(app)
    inference_queue.init_app(app)

    from app.api.auth import auth_bp
    from app.api.dataset import dataset_bp
    from app.api.detection import detection_bp
    from app.api.model import model_bp
    from app.api.report import report_bp
    from app.api.dashboard import dashboard_bp
    from app.api.image import image_bp
    from app.api.infer import infer_bp
    from app.api.rule import rule_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(dataset_bp, url_prefix="/api/dataset")
    app.register_blueprint(detection_bp, url_prefix="/api/detection")
    app.register_blueprint(model_bp, url_prefix="/api/model")
    app.register_blueprint(report_bp, url_prefix="/api/report")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(image_bp, url_prefix="/api/image")
    app.register_blueprint(infer_bp, url_prefix="/api/infer")
    app.register_blueprint(rule_bp, url_prefix="/api/rule")

    initialize_database(app)

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    return app
