import os

basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-secret-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES = 86400

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    UPLOAD_FOLDER = os.path.join(project_root, "uploads")
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

    CHECKPOINT_FOLDER = os.path.join(project_root, "checkpoints")
    ENGINE_PATH = os.path.join(project_root, "engine")
    ALGORITHM_RUNTIME_CONFIG = os.path.join(project_root, "instance", "algorithm_runtime.json")
    ALGORITHM_SERVICES_ROOT = os.path.join(project_root, "algorithms")

    AUTO_INIT_DB = os.environ.get("AUTO_INIT_DB", "true")
    AUTO_SYNC_DB_COLUMNS = os.environ.get("AUTO_SYNC_DB_COLUMNS", "true")
    AUTO_SEED_DB = os.environ.get("AUTO_SEED_DB", "true")

    # CPU-bound inference is deliberately serialized to avoid resource thrash.
    INFERENCE_WORKERS = int(os.environ.get("INFERENCE_WORKERS", "1"))
    INFERENCE_QUEUE_SIZE = int(os.environ.get("INFERENCE_QUEUE_SIZE", "8"))


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root@127.0.0.1:3306/scene_behavior",
    )
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "memory://")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "cache+memory://")


class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    REDIS_URL = os.environ.get("REDIS_URL")
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    SECRET_KEY = os.environ.get("SECRET_KEY")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
