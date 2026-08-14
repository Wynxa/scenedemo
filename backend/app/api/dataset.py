import os
from flask import Blueprint, request, current_app
from PIL import Image

from app.extensions import db
from app.models.dataset import Dataset, DatasetImage
from app.utils.response import success, error, table_data
from app.utils.auth import login_required, get_current_user_id
from app.utils.file_utils import save_uploaded_file

dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/list", methods=["GET"])
@login_required
def list_datasets():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)
    name = request.args.get("name", "")

    query = Dataset.query
    if name:
        query = query.filter(Dataset.name.contains(name))
    query = query.order_by(Dataset.create_time.desc())

    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data(
        rows=[d.to_dict() for d in pagination.items],
        total=pagination.total,
    )


@dataset_bp.route("", methods=["POST"])
@login_required
def create_dataset():
    data = request.get_json()
    ds = Dataset(
        name=data["name"],
        description=data.get("description", ""),
        creator_id=get_current_user_id(),
    )
    db.session.add(ds)
    db.session.commit()
    return success(ds.to_dict(), msg="创建成功")


@dataset_bp.route("/<int:ds_id>", methods=["GET"])
@login_required
def get_dataset(ds_id):
    ds = Dataset.query.get_or_404(ds_id)
    return success(ds.to_dict())


@dataset_bp.route("/<int:ds_id>", methods=["DELETE"])
@login_required
def delete_dataset(ds_id):
    ds = Dataset.query.get_or_404(ds_id)
    # Delete uploaded files
    for img in ds.images:
        full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], img.file_path)
        if os.path.exists(full_path):
            os.remove(full_path)
    db.session.delete(ds)
    db.session.commit()
    return success(msg="删除成功")


@dataset_bp.route("/<int:ds_id>/upload", methods=["POST"])
@login_required
def upload_images(ds_id):
    ds = Dataset.query.get_or_404(ds_id)
    files = request.files.getlist("files") or request.files.getlist("file")

    uploaded = 0
    for f in files:
        rel_path = save_uploaded_file(f)
        if not rel_path:
            continue

        full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], rel_path)
        try:
            img = Image.open(full_path)
            w, h = img.size
        except Exception:
            w, h = 0, 0

        file_size = os.path.getsize(full_path)
        dimg = DatasetImage(
            dataset_id=ds.id,
            file_path=rel_path,
            file_name=f.filename,
            width=w,
            height=h,
            file_size=file_size,
        )
        db.session.add(dimg)
        uploaded += 1

    ds.image_count = ds.images.count()
    db.session.commit()
    return success({"uploaded": uploaded}, msg=f"成功上传 {uploaded} 张图片")


@dataset_bp.route("/<int:ds_id>/images", methods=["GET"])
@login_required
def list_images(ds_id):
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 20, type=int)

    query = DatasetImage.query.filter_by(dataset_id=ds_id).order_by(DatasetImage.upload_time.desc())
    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data(
        rows=[img.to_dict() for img in pagination.items],
        total=pagination.total,
    )
