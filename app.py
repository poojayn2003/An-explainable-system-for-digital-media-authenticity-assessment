# Trustify AI - Real ML Backend
# Run: python app.py
# Exposes:
#   POST /api/analyze
#   GET  /api/health

import hashlib
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference.metadata import extract_metadata
from inference.predict import analyze_image


# --------------------------------------------------
# Project paths
# --------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent

# Grad-CAM output directory
GRADCAM_DIR = REPO_DIR / "outputs" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

app = Flask(__name__)

# Allow frontend requests from localhost
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

    prediction = result.get("prediction")

    if prediction == "REAL":
        return "original", "Original"

    if prediction == "AI-GENERATED":
        return "generated", "AI Generated"

    if prediction == "AI-EDITED / MANIPULATED":
        return "edited", "AI Edited"

    # Safe fallback
    return "original", str(prediction or "Unknown")


def build_reasons(result):
    """
    Convert the real ML decision into frontend-friendly
    explanation text.

    These reasons are based on the actual CIFAKE and CASIA
    scores returned by inference.predict.
    """

    reasons = []

    # --------------------------------------------------
    # Main decision reason
    # --------------------------------------------------

    reason = result.get("reason")

    if reason:
        reasons.append(reason)

    # --------------------------------------------------
    # CIFAKE signal
    # --------------------------------------------------

    cifake_fake = result.get("cifake_score")

    if cifake_fake is not None:

        cifake_fake = float(cifake_fake)
        cifake_real = 100.0 - cifake_fake

        if cifake_fake >= 50:
            reasons.append(
                f"AI-generation detector produced a "
                f"{cifake_fake:.2f}% AI-generation signal."
            )
        else:
            reasons.append(
                f"AI-generation detector produced a "
                f"{cifake_real:.2f}% real-image signal."
            )

    # --------------------------------------------------
    # CASIA signal
    # --------------------------------------------------

    casia_edited = result.get("casia_score")

    if casia_edited is not None:

        casia_edited = float(casia_edited)
        casia_real = 100.0 - casia_edited

        if casia_edited >= 50:
            reasons.append(
                f"Image-manipulation detector produced a "
                f"{casia_edited:.2f}% manipulation signal."
            )
        else:
            reasons.append(
                f"Image-manipulation detector produced a "
                f"{casia_real:.2f}% real-image signal."
            )

    # --------------------------------------------------
    # Decision-engine evidence
    # --------------------------------------------------

    evidence = result.get("evidence")

    if isinstance(evidence, list):
        for item in evidence:
            if item and item not in reasons:
                reasons.append(str(item))

    # Make sure frontend always receives an array
    if not reasons:
        reasons.append(
            "The authenticity assessment was completed successfully."
        )

    return reasons


def get_heatmap_filename(path_value):
    """
    Extract only the filename from a Grad-CAM path.

    The ML backend may return a Windows absolute path such as:
        C:\\...\\outputs\\gradcam\\image_cifake_gradcam.jpg

    The browser should receive only the filename through
    /outputs/gradcam/<filename>.
    """

    if not path_value:
        return None

    path = Path(str(path_value))

    # Path.name works correctly for normal paths.
    # Replace backslashes first so Windows paths are also handled
    # when returned as strings.
    filename = str(path).replace("\\", "/").split("/")[-1]

    if not filename:
        return None

    return filename


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    # --------------------------------------------------
    # Check upload
    # --------------------------------------------------

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded. Expected form field 'file'."
        }), 400

    uploaded = request.files["file"]

    if uploaded.filename == "":
        return jsonify({
            "error": "Empty filename."
        }), 400

    # --------------------------------------------------
    # Validate extension
    # --------------------------------------------------

    extension = Path(uploaded.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": (
                "Unsupported file type. "
                "Allowed: JPG, JPEG, PNG, WEBP, TIF, TIFF."
            )
        }), 400

    try:

        # --------------------------------------------------
        # Read uploaded image
        # --------------------------------------------------

        file_bytes = uploaded.read()

        if not file_bytes:
            return jsonify({
                "error": "Uploaded file is empty."
            }), 400

        # --------------------------------------------------
        # SHA-256 hash
        # --------------------------------------------------

        file_hash = hashlib.sha256(file_bytes).hexdigest().upper()

        # --------------------------------------------------
        # Create temporary image file
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_path = Path(temp_file.name)
            temp_file.write(file_bytes)

        try:

            # --------------------------------------------------
            # Run REAL Trustify AI ML inference
            # --------------------------------------------------

            ml_result = analyze_image(str(temp_path))

            metadata = extract_metadata(
                str(temp_path),
                original_filename=uploaded.filename
            )
            metadata = extract_metadata(str(temp_path))

        finally:

            # --------------------------------------------------
            # Remove temporary uploaded image
            # --------------------------------------------------

            if temp_path.exists():
                temp_path.unlink()

        # --------------------------------------------------
        # Convert ML result to frontend format
        # --------------------------------------------------

        classification, label = get_classification(ml_result)

        reasons = build_reasons(ml_result)

        # --------------------------------------------------
        # Confidence / trust score
        # --------------------------------------------------

        confidence = float(ml_result.get("confidence", 0))
        trust_score = int(round(float(ml_result.get("trust_score", 0))))

        # Keep values inside valid frontend ranges
        confidence = max(0.0, min(100.0, confidence))
        trust_score = max(0, min(100, trust_score))

        # --------------------------------------------------
        # Grad-CAM paths
        # --------------------------------------------------

        cifake_heatmap_filename = get_heatmap_filename(
            ml_result.get("cifake_heatmap")
        )

        casia_heatmap_filename = get_heatmap_filename(
            ml_result.get("casia_heatmap")
        )

        # --------------------------------------------------
        # Browser-accessible Grad-CAM URLs
        # --------------------------------------------------

        cifake_heatmap_url = None
        casia_heatmap_url = None

        if cifake_heatmap_filename:
            cifake_heatmap_url = (
                "/outputs/gradcam/" + cifake_heatmap_filename
            )

        if casia_heatmap_filename:
            casia_heatmap_url = (
                "/outputs/gradcam/" + casia_heatmap_filename
            )

        # --------------------------------------------------
        # Main heatmap used by the existing upload.html
        #
        # upload.html currently has only ONE heatmap image.
        # Prefer the heatmap associated with the final decision.
        # --------------------------------------------------

        prediction = ml_result.get("prediction")

        if prediction == "AI-GENERATED":
            heatmap_url = cifake_heatmap_url

        elif prediction == "AI-EDITED / MANIPULATED":
            heatmap_url = casia_heatmap_url

        else:
            # For REAL, use CIFAKE heatmap as the primary display.
            heatmap_url = cifake_heatmap_url or casia_heatmap_url

        # --------------------------------------------------
        # Frontend-compatible response
        # --------------------------------------------------

        response = {

            # ----------------------------------------------
            # Fields directly required by upload.html
            # ----------------------------------------------

            "classification": classification,
            "label": label,
            "confidence": confidence,
            "trust_score": trust_score,

            # Existing frontend displays the first 12 chars
            # in its report header.
            "file_hash": file_hash[:12],

            "reasons": reasons,
            "metadata": metadata,
            # ----------------------------------------------
            # Grad-CAM
            # ----------------------------------------------

            "heatmap_url": f"http://localhost:5000{heatmap_url}",
            "cifake_heatmap_url": f"http://localhost:5000{cifake_heatmap_url}",
            "casia_heatmap_url": f"http://localhost:5000{casia_heatmap_url}",

            # ----------------------------------------------
            # Real model-level signals
            # ----------------------------------------------

            "cifake_fake": (
                float(ml_result["cifake_score"])
                if ml_result.get("cifake_score") is not None
                else None
            ),

            "cifake_real": (
                100.0 - float(ml_result["cifake_score"])
                if ml_result.get("cifake_score") is not None
                else None
            ),

            "casia_edited": (
                float(ml_result["casia_score"])
                if ml_result.get("casia_score") is not None
                else None
            ),

            "casia_real": (
                100.0 - float(ml_result["casia_score"])
                if ml_result.get("casia_score") is not None
                else None
            ),

            # ----------------------------------------------
            # Original ML decision
            # ----------------------------------------------

            "prediction": ml_result.get("prediction"),
            "reason": ml_result.get("reason"),
            "explanation": ml_result.get("explanation"),

            # Keep evidence available for future frontend use
            "evidence": ml_result.get("evidence", []),
        }

        return jsonify(response)

    except Exception as e:

        print("\n" + "=" * 60)
        print("[Trustify AI] Inference error")
        print("=" * 60)
        print(str(e))
        print("=" * 60)

        return jsonify({
            "error": "Image analysis failed.",
            "details": str(e)
        }), 500


# --------------------------------------------------
# Grad-CAM static files
# --------------------------------------------------

@app.route("/outputs/gradcam/<path:filename>", methods=["GET"])
def serve_gradcam(filename):

    return send_from_directory(
        GRADCAM_DIR,
        filename
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

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

    print(f"Project:    {REPO_DIR}")
    print(f"Grad-CAM:   {GRADCAM_DIR}")
    print()
    print("API:        http://localhost:5000/api/analyze")
    print("Health:     http://localhost:5000/api/health")
    print("Grad-CAM:   http://localhost:5000/outputs/gradcam/<file>")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )