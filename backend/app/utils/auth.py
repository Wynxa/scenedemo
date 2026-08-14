"""JWT auth utilities using PyJWT directly."""

import functools
from datetime import datetime, timedelta

import jwt
from flask import request, current_app, g

from app.utils.response import unauthorized


def create_access_token(user_id):
    """Create a JWT access token for a user."""
    payload = {
        "sub": str(user_id),
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    secret = current_app.config.get("JWT_SECRET_KEY", "jwt-secret")
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token):
    """Decode and verify a JWT token. Returns payload or None."""
    secret = current_app.config.get("JWT_SECRET_KEY", "jwt-secret")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_token_from_header():
    """Extract Bearer token from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def login_required(fn=None, optional=False):
    """Decorator: require valid JWT token. Sets g.current_user_id."""

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            token = get_token_from_header()
            if not token:
                if optional:
                    g.current_user_id = None
                    return f(*args, **kwargs)
                return unauthorized()

            payload = decode_token(token)
            if not payload:
                return unauthorized("登录已过期，请重新登录")

            g.current_user_id = int(payload.get("sub", 0))
            return f(*args, **kwargs)

        return wrapper

    if fn is None:
        return decorator
    return decorator(fn)


def get_current_user_id():
    """Get current user ID from the g context."""
    return getattr(g, "current_user_id", None)
