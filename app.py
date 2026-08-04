# Trustify AI backend
# Run: pip install -r requirements.txt && python app.py
# Exposes POST /api/analyze. Returns a random placeholder result for now.

import random
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows upload.html (served from a different port/origin) to call this API

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# TODO: replace with the real TensorFlow/OpenCV/Grad-CAM pipeline
def run_inference(file_bytes: bytes, filename: str, content_type: str) -> dict:
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
