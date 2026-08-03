# Trustify AI

**Explainable AI-Based Digital Media & Identity Verification Platform**

Trustify AI helps users tell the difference between **Original**, **AI Edited**, and **AI Generated** media — and, unlike most detectors, explains *why* it reached that conclusion instead of returning a bare real/fake label.

---

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Verification Pipeline](#verification-pipeline)
- [API Contract](#api-contract)
- [Page Flow](#page-flow)
- [Roadmap](#roadmap)
- [Team](#team)

---

## Abstract

The rapid advancement of generative AI has made it possible to create highly realistic fake images, videos, audio clips, and even complete online identities. These synthetic media artifacts are increasingly used in misinformation campaigns, financial fraud, impersonation scams, and political manipulation. Existing detection tools typically classify content as "real" or "fake" without explanation, source verification, or identity-level trust analysis — which limits user confidence and makes informed decision-making difficult.

Trustify AI proposes an **Explainable AI-based digital media and identity verification platform** that combines image forensics, deepfake detection, audio-video synchronization checks, metadata examination, and behavioral consistency analysis into a single Trust Score — with a plain-language explanation behind every result.

## Problem Statement

- **Deepfakes** — face-swapped or reanimated video used to impersonate real people
- **Fake evidence** — synthetic photos/videos submitted in disputes or fraud claims
- **AI-generated images** — fully synthetic media passed off as real
- **Voice cloning** — cloned voices used in scam calls and impersonation
- **Misinformation** — fabricated media that spreads faster than any fact-check

Most existing detectors stop at a binary label with no reasoning, no confidence breakdown, and no report — which is not good enough when the decision actually matters.

## Key Features

- **Three-class classification** — Original / AI Edited / AI Generated, not just real-or-fake
- **Explainable AI** — every verdict comes with plain-language reasons, not just a score
- **Trust Score (0–100)** — a single fused score combining every signal checked
- **Investigation Report** — a downloadable summary of the full evidence trail
- **Multi-modal analysis** — image, video, and audio, plus metadata and persona-level consistency

## Project Structure

```
trustify-ai/
├── index.html          # Landing page
├── platform.html        # Full product story: problem, method, modules, roadmap
├── upload.html           # Upload → analysis → investigation report flow (frontend)
├── app.py                # Flask backend — POST /api/analyze
├── requirements.txt       # Backend dependencies
└── README.md
```

## Tech Stack

| Layer            | Technology                                  |
|-------------------|----------------------------------------------|
| Frontend          | HTML, CSS, JavaScript                        |
| Backend           | Python, Flask                                 |
| Model / Inference | TensorFlow                                    |
| Image processing  | OpenCV                                        |
| Explainability    | Grad-CAM                                      |

## Getting Started

### 1. Frontend (no build step required)

Open `index.html` directly in a browser, or serve the folder with any static server:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

### 2. Backend

```bash
pip install -r requirements.txt
python app.py
```

This starts the Flask API at `http://localhost:5000`.

> **Note:** `upload.html` will automatically try to reach the backend at `http://localhost:5000/api/analyze`. If the backend isn't running, it falls back to an on-page simulated result so the UI can still be demoed on its own — check the browser console for a warning when this happens.

### 3. Wiring in the real model

`app.py`'s `run_inference()` function currently returns a random placeholder result. Replace its internals with the real pipeline:

1. Load the trained TensorFlow model(s) once at startup.
2. Preprocess the uploaded file (OpenCV for image/video frames, an audio library for audio).
3. Run inference for artifact/deepfake detection.
4. Run Grad-CAM (or your chosen explainability method) and translate its output into the `reasons` list.
5. Fuse all signals into a single `trust_score` (0–100).

The JSON shape returned must stay the same — see [API Contract](#api-contract) — so the frontend needs no changes once the real model is in place.

## Verification Pipeline

Every submitted file moves through the same six stages before a verdict is shown:

1. **Evidence** — file is received, encrypted, and fingerprinted
2. **Metadata Analysis** — EXIF, timestamps, and edit history are checked
3. **AI Artifact Detection** — GAN/diffusion fingerprints, splice traces, synthetic patterns
4. **Explainability Engine** — signals are translated into plain-language reasons
5. **Trust Score** — all signals are fused into a single 0–100 score
6. **Investigation Report** — a complete, downloadable summary is generated

## API Contract

**`POST /api/analyze`**

Request: `multipart/form-data` with a single field `file`.

Response:

```json
{
  "classification": "original",
  "label": "Original",
  "confidence": 97.3,
  "trust_score": 84,
  "reasons": [
    "Sensor noise pattern matches expected camera profile",
    "No GAN or diffusion artifacts found in facial or background regions"
  ],
  "file_hash": "9F2AC31E7B4D"
}
```

`classification` must be one of `original`, `edited`, or `generated` — the frontend uses this exact value to color and label the verdict.

## Page Flow

```
index.html  (Landing)
    │
    ├── Learn More  ──────────►  platform.html  (Full product story)
    │
    └── Start Verification ───►  upload.html  (Upload → Analyze → Report)
                                       │
                                       ▼
                                  Flask backend
                                  POST /api/analyze
```

## Roadmap

- Browser extension for in-page verification
- Public API for third-party integrations
- Mobile app
- Blockchain-based provenance tracking
- Real-time / live-stream verification

## Team

| Name          | Role                | Reg. No. |
|----------------|----------------------|-----------|
| _Pooja Y N_   | _Frontend / Backend_ | _1JB24CD403_       |
| _Amrutha R N_  | _Backend_                | _1JB23CD006_       |
| _Pooja A R_  | _ML_                | _1JB23CD037_       |

**Guide:** _Vindyashree_, Department of Computer Science & Engineering (Data Science)
**Institution:** _SJBIT_ — Academic Year 2026–27

---

*This is an academic project. The detection results shown are currently simulated pending integration of the trained model described above.*
