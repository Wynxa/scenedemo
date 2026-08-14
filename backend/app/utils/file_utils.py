import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from app.config import Config


def allowed_image(filename):
    """Check if file extension is allowed."""
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in Config.ALLOWED_IMAGE_EXTENSIONS


def save_uploaded_file(file, subfolder="images"):
    """Save uploaded file with UUID name. Returns relative path."""
    if not file or not allowed_image(file.filename):
        return None

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"

    today = datetime.now().strftime("%Y/%m/%d")
    dest_dir = os.path.join(Config.UPLOAD_FOLDER, subfolder, today)
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, unique_name)
    file.save(dest_path)

    return os.path.join(subfolder, today, unique_name).replace("\\", "/")
