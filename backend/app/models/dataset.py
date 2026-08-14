from datetime import datetime
from app.extensions import db


class Dataset(db.Model):
    __tablename__ = "dataset"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    image_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(16), default="active")  # active / archived
    creator_id = db.Column(db.Integer, default=0)
    create_time = db.Column(db.DateTime, default=datetime.now)

    images = db.relationship("DatasetImage", backref="dataset", lazy="dynamic", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "imageCount": self.image_count,
            "status": self.status,
            "creatorId": self.creator_id,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
        }


class DatasetImage(db.Model):
    __tablename__ = "dataset_image"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = db.Column(db.String(512), nullable=False)
    file_name = db.Column(db.String(256))
    width = db.Column(db.Integer, default=0)
    height = db.Column(db.Integer, default=0)
    file_size = db.Column(db.BigInteger, default=0)
    upload_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "datasetId": self.dataset_id,
            "filePath": self.file_path,
            "fileName": self.file_name,
            "width": self.width,
            "height": self.height,
            "fileSize": self.file_size,
            "uploadTime": self.upload_time.strftime("%Y-%m-%d %H:%M:%S") if self.upload_time else None,
        }
