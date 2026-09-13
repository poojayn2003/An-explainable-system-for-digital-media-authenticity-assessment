"""
Trustify AI - Flask API

REST API for image authenticity analysis.
"""

import sys
from pathlib import Path

from flask import Flask, request, jsonify


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# =========================================================
# IMPORT MODEL LOADER
# =========================================================

from backend.model_loader import analyze_image


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "tif",
    "tiff"
}


def allowed_file(filename):
    """
    Check whether the uploaded file has an allowed extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Trustify AI API is running"
    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # Check image
    # -----------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "status": "error",
            "message": "No image file provided."
        }), 400

    image = request.files["image"]

    # -----------------------------------------------------
    # Check filename
    # -----------------------------------------------------

    if image.filename == "":

        return jsonify({
            "status": "error",
            "message": "No image selected."
        }), 400

    if not allowed_file(image.filename):

        return jsonify({
            "status": "error",
            "message": "Unsupported image format."
        }), 400

    # -----------------------------------------------------
    # Save temporary image
    # -----------------------------------------------------

    upload_dir = BASE_DIR / "backend" / "uploads"

    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = upload_dir / image.filename

    try:

        image.save(image_path)

        # -------------------------------------------------
        # Run model inference
        # -------------------------------------------------

        result = analyze_image(image_path)

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:

        # -------------------------------------------------
        # Delete temporary image
        # -------------------------------------------------

        if image_path.exists():

            image_path.unlink()


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )