"""
Initialize database manually.
This script reuses the same bootstrap logic as app startup.
Usage: python init_db.py
"""

from app import create_app
from app.services.db_bootstrap import initialize_database


def init() -> None:
    app = create_app()
    initialize_database(app)
    print("Database initialized successfully.")


if __name__ == "__main__":
    init()
