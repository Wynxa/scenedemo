from datetime import datetime
from app.extensions import db


class User(db.Model):
    __tablename__ = "sys_user"

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password = db.Column(db.String(256), nullable=False)
    real_name = db.Column(db.String(64))
    email = db.Column(db.String(128))
    phone = db.Column(db.String(20))
    avatar = db.Column(db.String(256))
    role = db.Column(db.String(32), default="inspector")  # admin / inspector / viewer
    dept_id = db.Column(db.Integer, default=0)
    status = db.Column(db.String(1), default="0")  # 0=active, 1=disabled
    create_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "userId": self.user_id,
            "username": self.username,
            "realName": self.real_name,
            "email": self.email,
            "phone": self.phone,
            "avatar": self.avatar,
            "role": self.role,
            "deptId": self.dept_id,
            "status": self.status,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }
