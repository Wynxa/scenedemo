"""
Celery worker entry point.
Usage: celery -A celery_worker worker --loglevel=info --pool=solo
"""

import os
from app import create_app
from app.extensions import celery

app = create_app(os.environ.get("FLASK_ENV", "development"))
