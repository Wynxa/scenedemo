"""
Development server with SocketIO support.
Database bootstrap is handled inside create_app().
Usage: python run.py
"""

from app import create_app
from app.extensions import socketio

app = create_app()

if __name__ == "__main__":
    # Scene-graph initialisation writes runtime support files under backend/tmp.
    # Do not let Werkzeug mistake those generated files for source changes and
    # restart the server while an inference worker is using it.
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
