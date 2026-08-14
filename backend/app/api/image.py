import os
from flask import Blueprint, current_app, request
from PIL import Image

from app.extensions import db
from app.models.inference import ImageAsset
from app.services.inference_service import InferenceOrchestrator
from app.utils.auth import get_current_user_id, login_required
from app.utils.file_utils import save_uploaded_file
from app.utils.response import error, success, table_data

image_bp = Blueprint("image", __name__)


@image_bp.route("/upload", methods=["POST"])
@login_required
def upload_image():
    if "file" not in request.files:
        return error("请上传图片", 400)

    file = request.files["file"]
    rel_path = save_uploaded_file(file, subfolder="pipeline_images")
    if not rel_path:
        return error("不支持的图片格式", 400)

    full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], rel_path)
    try:
        img = Image.open(full_path)
        width, height = img.size
    except Exception:
        width, height = 0, 0

    asset = ImageAsset(
        dataset_id=request.form.get("datasetId", type=int),
        source_type=request.form.get("sourceType", "upload"),
        file_name=file.filename,
        file_path=rel_path,
        file_url=f"/uploads/{rel_path}",
        width=width,
        height=height,
        file_size=os.path.getsize(full_path),
        md5=InferenceOrchestrator.compute_file_md5(full_path),
        creator_id=get_current_user_id(),
    )
    db.session.add(asset)
    db.session.commit()
    return success(asset.to_dict(), msg="图片上传成功")


@image_bp.route("/list", methods=["GET"])
@login_required
def list_images():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)
    query = ImageAsset.query.order_by(ImageAsset.create_time.desc())
    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data([row.to_dict() for row in pagination.items], pagination.total)


@image_bp.route("/<int:image_id>", methods=["GET"])
@login_required
def get_image(image_id):
    asset = ImageAsset.query.get_or_404(image_id)
    return success(asset.to_dict())
