# An Explainable System for Digital Media Authenticity Assessment

## Trustify AI

**Trustify AI** is an explainable AI-based image authenticity assessment system designed to distinguish between **Original**, **AI Edited / Manipulated**, and **AI Generated** images.

Instead of providing only a real/fake prediction, the system provides supporting evidence through **AI-based classification, confidence scores, metadata analysis, SHA-256 file fingerprinting, and Grad-CAM visual explanations**.

> **Current implementation scope:** Image authenticity assessment. Video, audio, identity verification, and live-stream analysis are outside the scope of the current implementation.

---

## Table of Contents

* [Abstract](#abstract)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Project Structure](#project-structure)
* [Tech Stack](#tech-stack)
* [Getting Started](#getting-started)
* [Verification Pipeline](#verification-pipeline)
* [Classification Logic](#classification-logic)
* [API Contract](#api-contract)
* [Page Flow](#page-flow)
* [Models and Datasets](#models-and-datasets)
* [Explainability](#explainability)
* [Evaluation and Training](#evaluation-and-training)
* [Roadmap](#roadmap)
* [Team](#team)
* [Important Notes](#important-notes)

---

## Abstract

The rapid development of generative AI has made it increasingly difficult to distinguish authentic images from AI-generated and manipulated content. Images can be synthetically generated, digitally edited, or manipulated in ways that may not be obvious to human observers.

Traditional detection systems often provide only a binary **real/fake** result, making it difficult for users to understand why a particular image was classified as suspicious.

**Trustify AI** addresses this problem through an explainable image-authenticity assessment pipeline. The current implementation uses two trained **EfficientNet-B0** models:

1. **CIFAKE model** — identifies whether an image is likely to be AI-generated.
2. **CASIA model** — identifies whether an image shows signs of manipulation or editing.

The outputs are processed by a decision engine to produce one of three classifications:

* **Original**
* **AI Edited / Manipulated**
* **AI Generated**

The system additionally performs metadata analysis and generates **Grad-CAM heatmaps** to provide visual evidence about the regions that influenced the model predictions.

---

## Problem Statement

AI-generated and digitally manipulated images are increasingly used for:

* Misinformation and fabricated content
* Fraudulent evidence
* Online impersonation
* Misleading social-media content
* Unauthorized image manipulation
* Synthetic media distribution

A simple real/fake classification does not provide enough information for users to understand the basis of a detection result.

Therefore, this project aims to develop an **explainable image authenticity assessment system** that combines machine-learning predictions with forensic evidence and visual explanations.

---

## Objectives

The major objectives of Trustify AI are:

* Detect AI-generated images.
* Detect digitally manipulated or edited images.
* Distinguish suspicious images from likely original images.
* Provide confidence-based predictions.
* Analyze available image metadata as supporting evidence.
* Generate Grad-CAM visualizations for model explainability.
* Provide human-readable reasons behind the classification.
* Generate an investigation-style report through the web interface.
* Provide a REST API for image analysis.

---

## Key Features

### 1. Three-Class Authenticity Assessment

The system classifies images into:

| Classification              | Meaning                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| **Original**                | No strong evidence of AI generation or manipulation was detected |
| **AI Edited / Manipulated** | The image shows evidence associated with digital manipulation    |
| **AI Generated**            | The image has a high probability of being AI-generated           |

### 2. Dual-Model Analysis

Two EfficientNet-B0 models are used for complementary detection tasks:

* **CIFAKE EfficientNet-B0** → AI-generated image detection
* **CASIA EfficientNet-B0** → image manipulation detection

### 3. Explainable AI

Grad-CAM is used to generate heatmaps showing image regions that contributed to the model's prediction.

### 4. Metadata Analysis

Available image metadata is extracted and presented as supporting forensic information.

Metadata is treated as **supporting evidence**, rather than as a direct classification rule.

### 5. Confidence and Trust Score

The system provides:

* Model confidence values
* A **0–100 Trust Score**

The current Trust Score is derived primarily from the model confidence/decision outcome. It should not be interpreted as a mathematically calibrated probability or as a guarantee that an image is authentic.

### 6. File Fingerprinting

Each uploaded image is assigned a **SHA-256 hash** that can be used as a digital fingerprint for the analyzed file.

### 7. Investigation Report

The frontend provides options to view and export the analysis results, including the classification, evidence, metadata, confidence values, and explainability information.

---

## System Architecture

The current production analysis flow is:

```text
                    Uploaded Image
                           │
                           ▼
                    Flask API
                 POST /api/analyze
                           │
                           ▼
                    Image Validation
                           │
                           ├──────────────► SHA-256 File Hash
                           │
                           ├──────────────► Metadata Extraction
                           │
                           ▼
                    Image Preprocessing
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
        CIFAKE EfficientNet-B0   CASIA EfficientNet-B0
                 │                   │
                 ▼                   ▼
        AI-Generated Score      Manipulation Score
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    Decision Engine
                           │
                           ▼
              Original / AI Edited /
                  AI Generated
                           │
                           ├──────────────► Grad-CAM
                           │
                           ├──────────────► Evidence
                           │
                           └──────────────► Trust Score
                                      │
                                      ▼
                               JSON Response
                                      │
                                      ▼
                              Frontend Report
```

---

## Project Structure

```text
trustify-ai/
│
├── index.html
├── platform.html
├── upload.html
├── about.html
├── contact.html
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── app.py
│   └── model_loader.py
│
├── inference/
│   ├── predict.py
│   ├── predict_casia.py
│   ├── decision_engine.py
│   ├── metadata.py
│   ├── gradcam.py
│   ├── gradcam_casia.py
│   ├── fusion.py
│   ├── calibrate_casia.py
│   ├── evaluate_fusion.py
│   ├── test_fusion.py
│   └── test_real_world.py
│
├── preprocessing/
│   ├── clean_casia.py
│   ├── download_real_images.py
│   ├── prepare_casia.py
│   └── prepare_realworld_finetune.py
│
├── training/
│   ├── train.py
│   ├── train_casia.py
│   └── finetune_realworld.py
│
└── evaluation/
    └── evaluate.py
```

---

## Tech Stack

| Layer                | Technology                       |
| -------------------- | -------------------------------- |
| Frontend             | HTML, CSS, JavaScript            |
| Backend              | Python, Flask                    |
| API                  | REST API                         |
| Machine Learning     | PyTorch                          |
| CNN Architecture     | EfficientNet-B0                  |
| Image Processing     | Pillow, OpenCV                   |
| Numerical Processing | NumPy                            |
| Explainability       | Grad-CAM                         |
| Metadata             | Python image metadata extraction |
| Evaluation           | scikit-learn                     |
| Testing              | Pytest                           |
| Version Control      | Git / GitHub                     |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd trustify-ai
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The backend requires the dependencies used by the current inference and evaluation pipeline, including:

* Flask
* Flask-CORS
* PyTorch
* Torchvision
* Pillow
* NumPy
* OpenCV
* scikit-learn
* Requests

### 4. Add Trained Model Weights

The trained model weights are not included in the repository.

Place the required checkpoints at:

```text
models/
├── efficientnet_best.pth
└── casia_efficientnet_best.pth
```

The `.pth` files are excluded from Git through `.gitignore`.

### 5. Start the Backend

From the project root:

```bash
python app.py
```

The Flask server runs at:

```text
http://localhost:5000
```

### 6. Start the Frontend

The frontend can be served using a simple local HTTP server:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

Navigate to **Start Verification** to use the image-analysis interface.

---

## Verification Pipeline

### Step 1 — File Validation

The Flask backend receives the uploaded image and validates:

* File presence
* File extension
* File size
* Image readability

The current maximum upload size is **50 MB**.

Supported formats include:

```text
JPG
JPEG
PNG
WEBP
TIF
TIFF
```

### Step 2 — File Fingerprinting

A SHA-256 hash is generated for the uploaded file.

This provides a unique fingerprint for the analyzed file and allows the file to be identified later based on its content.

### Step 3 — Image Preprocessing

The image is loaded and converted into the format required by the EfficientNet-B0 inference pipeline.

The model input is resized/preprocessed to:

```text
224 × 224 × 3
```

### Step 4 — CIFAKE Analysis

The CIFAKE EfficientNet-B0 model estimates whether the image is likely to be:

* AI-generated
* Real

The resulting confidence is passed to the decision engine.

### Step 5 — CASIA Analysis

The CASIA EfficientNet-B0 model evaluates whether the image contains evidence associated with manipulation/editing.

Its output is used to identify potential:

* Edited images
* Manipulated regions
* Forgery-related patterns

### Step 6 — Decision Engine

The two model outputs are evaluated using predefined thresholds.

### Step 7 — Explainability

Grad-CAM generates heatmaps for the model predictions.

These heatmaps help visualize which regions of the image contributed to the model's decision.

### Step 8 — Metadata Analysis

Available image metadata is extracted and returned as additional forensic evidence.

### Step 9 — Final Result

The backend returns the classification, confidence information, trust score, metadata, evidence, and heatmap URLs to the frontend.

---

## Classification Logic

The current decision engine uses the following thresholds:

```text
CIFAKE AI-generated threshold = 0.80
CASIA manipulation threshold  = 0.70
```

The classification logic is:

```text
IF CIFAKE score >= 0.80
        │
        ▼
   AI GENERATED

ELSE IF CASIA score >= 0.70
        │
        ▼
   AI EDITED / MANIPULATED

ELSE
        │
        ▼
      ORIGINAL
```

This means the current system uses **two binary model predictions followed by a rule-based decision engine**, rather than a single neural network directly trained for three-class classification.

---

## API Contract

### Health Check

**GET**

```text
/api/health
```

Example response:

```json
{
  "status": "ok",
  "service": "Trustify AI",
  "models": [
    "CIFAKE EfficientNet-B0",
    "CASIA EfficientNet-B0"
  ]
}
```

---

### Image Analysis

**POST**

```text
/api/analyze
```

Request:

```text
multipart/form-data
```

Required field:

```text
file
```

Example using cURL:

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@sample.jpg"
```

Example using Python:

```python
import requests

with open("sample.jpg", "rb") as image:
    response = requests.post(
        "http://localhost:5000/api/analyze",
        files={"file": image}
    )

print(response.json())
```

### Response

A successful response contains fields such as:

```json
{
  "classification": "original",
  "label": "Original",
  "confidence": 97.3,
  "trust_score": 97.3,
  "file_hash": "9f2ac31e7b4d...",
  "reasons": [],
  "metadata": {},
  "heatmap_url": "/outputs/...",
  "cifake_heatmap_url": "/outputs/...",
  "casia_heatmap_url": "/outputs/...",
  "cifake_fake": 0.12,
  "cifake_real": 0.88,
  "casia_edited": 0.15,
  "casia_real": 0.85,
  "prediction": "Original",
  "reason": "...",
  "explanation": "...",
  "evidence": []
}
```

The exact confidence, evidence, metadata, and heatmap values depend on the uploaded image and the loaded model checkpoints.

---

## Page Flow

```text
                    index.html
                  Landing Page
                       │
          ┌────────────┴────────────┐
          │                         │
      Learn More              Start Verification
          │                         │
          ▼                         ▼
   platform.html              upload.html
   Product Details             Upload Image
                                      │
                                      ▼
                               Flask Backend
                                      │
                                      ▼
                              POST /api/analyze
                                      │
                                      ▼
                              Model Inference
                                      │
                                      ▼
                              Decision + Evidence
                                      │
                                      ▼
                             Investigation Report
```

Additional pages include:

* `about.html` — project information
* `contact.html` — contact page
* `platform.html` — platform methodology and product information

---

## Models and Datasets

### CIFAKE

The CIFAKE dataset is used for AI-generated image detection.

The project uses an **EfficientNet-B0** architecture trained for binary classification between real and AI-generated images.

### CASIA 2.0

The CASIA 2.0 dataset is used for image manipulation/forgery detection.

A separate **EfficientNet-B0** model is trained to identify manipulated versus authentic images.

### Model Checkpoints

The trained weights are expected at:

```text
models/efficientnet_best.pth
models/casia_efficientnet_best.pth
```

These files are intentionally excluded from version control because of their size.

---

## Explainability

Trustify AI uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations for model predictions.

Instead of showing only:

```text
AI Generated — 94%
```

the system can also produce a heatmap indicating the image regions that contributed strongly to the model's prediction.

The current implementation generates separate heatmaps for the CIFAKE and CASIA analysis pipelines.

> Grad-CAM highlights regions that influenced the model prediction. It should not be interpreted as definitive proof that a particular region was manipulated.

---

## Metadata Analysis

The system extracts available image metadata where present.

Potential metadata information includes:

* Image format
* Dimensions
* EXIF information
* Camera-related information
* Timestamp-related information
* Software/editing information

Metadata availability depends on the uploaded image. Many platforms remove or modify metadata when images are uploaded or reprocessed.

Therefore, metadata is treated as **supporting forensic evidence** and does not independently determine the final classification.

---

## Evaluation and Training

The repository includes scripts for model training, preprocessing, evaluation, and experimentation.

### Training

```text
training/train.py
training/train_casia.py
training/finetune_realworld.py
```

### Preprocessing

```text
preprocessing/clean_casia.py
preprocessing/download_real_images.py
preprocessing/prepare_casia.py
preprocessing/prepare_realworld_finetune.py
```

### Evaluation

```text
evaluation/evaluate.py
```

Additional inference and experimental evaluation scripts are available under:

```text
inference/
```

---

## Experimental Fusion Module

The repository also contains:

```text
inference/fusion.py
inference/evaluate_fusion.py
inference/test_fusion.py
```

These files represent an **experimental/alternative fusion approach** for combining forensic signals.

They should not be confused with the primary production decision path currently used by:

```text
app.py
    ↓
inference/predict.py
    ↓
inference/decision_engine.py
```

The current production classification is based on the CIFAKE and CASIA model outputs and their predefined decision thresholds.

---

## Security and File Handling

The backend:

* Validates uploaded files.
* Limits the maximum upload size.
* Generates a SHA-256 fingerprint.
* Uses temporary files during analysis.
* Removes the temporary uploaded file after processing.

> The current implementation does **not** provide encryption of uploaded files. Encryption and stronger production-grade storage/security controls can be added in future versions.

---

## Roadmap

Future improvements may include:

* Improved calibration of model confidence
* More robust real-world image testing
* Additional forensic models
* Better multimodal evidence fusion
* Video authenticity analysis
* Audio deepfake detection
* Browser extension for image verification
* Public API for third-party integrations
* Mobile application
* Provenance and content-credential integration
* Real-time media verification

Video and audio analysis are considered future extensions and are **not part of the current implementation**.

---


## Team

| Name            | Role               | Reg. No.   |
| --------------- | ------------------ | ---------- |
| **Pooja Y N**   | Frontend / Backend | 1JB24CD403 |
| **Amrutha R N** | Backend            | 1JB23CD006 |
| **Pooja A R**   | Machine Learning   | 1JB23CD037 |

### Guide

**Vindyashree**
Department of Computer Science & Engineering (Data Science)

### Institution

**SJB Institute of Technology (SJBIT)**

**Academic Year:** 2026–27

---

## Important Notes

* This is an **academic project** and should not be treated as a definitive forensic or legal authentication service.
* Detection accuracy depends on the quality, diversity, and limitations of the trained models and datasets.
* A high confidence score does not guarantee that an image is authentic or fake.
* Metadata can be missing, modified, or removed during image processing.
* Grad-CAM provides model-attribution information rather than absolute proof of manipulation.
* The current implementation focuses on **image authenticity assessment**.
* Video, audio, identity verification, behavioral analysis, and live-stream verification are not implemented in the current version.
* Model checkpoint files (`.pth`) and datasets are not included in the Git repository.

---

## Project Status

**Current status:** Working image-analysis backend with trained-model inference, dual-model decision logic, metadata extraction, Grad-CAM explainability, and frontend investigation/report flow.

**Primary API:** `POST /api/analyze`

**Models:** CIFAKE EfficientNet-B0 + CASIA EfficientNet-B0

**Output:** Original / AI Edited / AI Generated

---

*An academic project developed at SJB Institute of Technology for the 2026–27 academic year.*
