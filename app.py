"""
Trustify AI — backend stub

Run:
    pip install -r requirements.txt
    python app.py

This exposes POST /api/analyze which upload.html calls with the uploaded file.
Right now it returns a RANDOM placeholder verdict so you can test the full
pipeline end-to-end before your real model is ready. Replace `run_inference()`
with your actual TensorFlow / OpenCV / Grad-CAM pipeline — the JSON shape it
returns is the contract the frontend already expects, so nothing else needs
to change once you swap this function's internals.
"""

import random
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows upload.html (served from a different port/origin) to call this API

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ---------------------------------------------------------------------------
# Response contract (what upload.html's JS expects back):
#
# {
#   "classification": "original" | "edited" | "generated",
#   "label": "Original" | "AI Edited" | "AI Generated",
#   "confidence": 97.3,             # float, 0-100
#   "trust_score": 84,              # int, 0-100
#   "reasons": ["...", "...", ...]  # plain-language explanation bullets
# }
# ---------------------------------------------------------------------------


def run_inference(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """
    TODO: replace this entire function body with your real pipeline:

      1. Save/stream `file_bytes` to a temp path or load directly with OpenCV
         (cv2.imdecode) / a video reader for video / librosa for audio.
      2. Run your metadata check (EXIF via Pillow/exifread).
      3. Run your trained TensorFlow model(s) for GAN/deepfake artifact
         detection — model.predict(preprocessed_input).
      4. Run Grad-CAM (or your explainability method of choice) to produce
         the highlighted-region heatmap and turn its output into the
         plain-language `reasons` list.
      5. Fuse the signals into a single 0-100 trust_score however your
         project defines that fusion (weighted average, learned model, etc).

    For now this just returns a random plausible-looking result so the
    frontend can be fully tested before the real model exists.
    """
    classes = [
        {
            "classification": "original",
            "label": "Original",
            "reasons": [
                "Sensor noise pattern matches expected camera profile",
                "No GAN or diffusion artifacts found in facial or background regions",
                "Metadata timestamps are internally consistent",
                "Compression history matches a single capture-and-share cycle",
            ],
            "trust_range": (82, 97),
        },
        {
            "classification": "edited",
            "label": "AI Edited",
            "reasons": [
                "Localized inpainting artifacts detected in one region",
                "Metadata shows an editing tool signature after capture",
                "Lighting direction is inconsistent between subject and background",
                "Compression pattern differs between edited and untouched regions",
            ],
            "trust_range": (35, 60),
        },
        {
            "classification": "generated",
            "label": "AI Generated",
            "reasons": [
                "GAN artifacts detected across facial region",
                "No sensor noise pattern found — inconsistent with any camera",
                "Metadata contains no capture device information",
                "Frequency-domain analysis shows synthetic generation signature",
            ],
            "trust_range": (5, 28),
        },
    ]

    pick = random.choice(classes)
    confidence = round(random.uniform(91, 99), 1)
    trust_score = random.randint(*pick["trust_range"])

    return {
        "classification": pick["classification"],
        "label": pick["label"],
        "confidence": confidence,
        "trust_score": trust_score,
        "reasons": pick["reasons"],
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Expected form field 'file'."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    file_bytes = uploaded.read()
    result = run_inference(file_bytes, uploaded.filename, uploaded.content_type)

    result["file_hash"] = hashlib.sha256(file_bytes).hexdigest()[:12].upper()
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
