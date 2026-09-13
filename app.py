# Trustify AI - Real ML Backend
# Run: python app.py
# Exposes POST /api/analyze

import hashlib
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# REPO_DIR = Path(__file__).resolve().parent
# ML_PROJECT_DIR = Path(r"C:\AIProjects\DigitalMediaAuthenticityAssessment")

# # Allow imports from the ML project
# sys.path.insert(0, str(ML_PROJECT_DIR))

# from inference.predict import analyze_image
REPO_DIR = Path(__file__).resolve().parent

from inference.predict import analyze_image


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_classification(result):
    """
    Convert the ML decision-engine prediction into
    the classification values expected by upload.html.
    """

    prediction = result["prediction"]

    if prediction == "REAL":
        return "original", "Original"

    if prediction == "AI-GENERATED":
        return "generated", "AI Generated"

    if prediction == "AI-EDITED / MANIPULATED":
        return "edited", "AI Edited"

    return "original", prediction


def build_reasons(result):
    """
    Convert the model decision into frontend-friendly
    explanation text.
    """

    reasons = []

    # Main decision reason
    if result.get("reason"):
        reasons.append(result["reason"])

    # CIFAKE signal
    cifake_fake = result.get("cifake_fake")
    cifake_real = result.get("cifake_real")

    if cifake_fake is not None and cifake_real is not None:
        if cifake_fake >= 50:
            reasons.append(
                f"AI-generation detector produced a {cifake_fake:.2f}% "
                f"fake-generation signal."
            )
        else:
            reasons.append(
                f"AI-generation detector produced a {cifake_real:.2f}% "
                f"real-image signal."
            )

    # CASIA signal
    casia_edited = result.get("casia_edited")
    casia_real = result.get("casia_real")

    if casia_edited is not None and casia_real is not None:
        if casia_edited >= 50:
            reasons.append(
                f"Image-manipulation detector produced a {casia_edited:.2f}% "
                f"manipulation signal."
            )
        else:
            reasons.append(
                f"Image-manipulation detector produced a {casia_real:.2f}% "
                f"real-image signal."
            )

    return reasons


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    # Check upload
    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded. Expected form field 'file'."
        }), 400

    uploaded = request.files["file"]

    if uploaded.filename == "":
        return jsonify({
            "error": "Empty filename."
        }), 400

    # Validate extension
    extension = Path(uploaded.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": (
                "Unsupported file type. "
                "Allowed: JPG, JPEG, PNG, WEBP, TIF, TIFF."
            )
        }), 400

    try:
        # Read uploaded image
        file_bytes = uploaded.read()

        if not file_bytes:
            return jsonify({
                "error": "Uploaded file is empty."
            }), 400

        # SHA-256 hash
        file_hash = hashlib.sha256(file_bytes).hexdigest().upper()

        # Create temporary image file
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_path = Path(temp_file.name)
            temp_file.write(file_bytes)

        try:
            # Run real Trustify AI inference
            ml_result = analyze_image(str(temp_path))

        finally:
            # Remove temporary image
            if temp_path.exists():
                temp_path.unlink()

        # Convert ML result to frontend format
        classification, label = get_classification(ml_result)

        reasons = build_reasons(ml_result)

        response = {
            "classification": classification,
            "label": label,
            "confidence": ml_result["confidence"],
            "trust_score": ml_result["trust_score"],
            "file_hash": file_hash[:12],
            "reasons": reasons,

            # Keep model-level signals available
            "cifake_fake": ml_result.get("cifake_fake"),
            "cifake_real": ml_result.get("cifake_real"),
            "casia_edited": ml_result.get("casia_edited"),
            "casia_real": ml_result.get("casia_real"),

            # Main backend decision
            "prediction": ml_result.get("prediction"),
            "reason": ml_result.get("reason"),
        }

        return jsonify(response)

    except Exception as e:

        print("\n[Trustify AI] Inference error:")
        print(str(e))

        return jsonify({
            "error": "Image analysis failed.",
            "details": str(e)
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Trustify AI",
        "models": [
            "CIFAKE EfficientNet-B0",
            "CASIA EfficientNet-B0"
        ]
    })


# --------------------------------------------------
# Start server
# --------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Trustify AI - Digital Media Authenticity System")
    print("=" * 60)
    #print(f"ML Project: {ML_PROJECT_DIR}")
    print(f"Project:     {REPO_DIR}")
    print("Endpoint:   http://localhost:5000/api/analyze")
    print("Health:     http://localhost:5000/api/health")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )