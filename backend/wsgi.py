"""
WSGI entry point for production deployment (gunicorn).
Usage: gunicorn wsgi:app --worker-class eventlet -w 1 --bind 0.0.0.0:5000
"""

import os
from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
