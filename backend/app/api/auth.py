from flask import Blueprint, request
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db
from app.models.user import User
from app.utils.response import success, error
from app.utils.auth import login_required, create_access_token, get_current_user_id

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username", "")
    password = data.get("password", "")

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return error("用户名或密码错误", 400)

    if user.status == "1":
        return error("账号已被禁用", 403)

    token = create_access_token(user.user_id)
    return success({
        "token": token,
        "user": user.to_dict(),
    }, msg="登录成功")


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    return success(msg="退出成功")


@auth_bp.route("/userinfo", methods=["GET"])
@login_required
def userinfo():
    user_id = get_current_user_id()
    user = User.query.get(user_id)
    if not user:
        return error("用户不存在", 404)

    return success({
        "user": user.to_dict(),
        "roles": [user.role],
        "permissions": ["*"],
    })


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    existing = User.query.filter_by(username=data["username"]).first()
    if existing:
        return error("用户名已存在", 400)

    user = User(
        username=data["username"],
        password=bcrypt.hash(data["password"]),
        real_name=data.get("realName", ""),
        email=data.get("email", ""),
        role=data.get("role", "inspector"),
    )
    db.session.add(user)
    db.session.commit()
    return success(user.to_dict(), msg="注册成功")
