"""
Trustify AI - Model Loader

Loads the trained CIFAKE and CASIA EfficientNet-B0 models
once when the Flask application starts.

Provides image analysis for the REST API.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from inference.decision_engine import decide


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CIFAKE_MODEL_PATH = (
    BASE_DIR / "models" / "efficientnet_best.pth"
)

CASIA_MODEL_PATH = (
    BASE_DIR / "models" / "casia_efficientnet_best.pth"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# MODEL CREATION
# =========================================================

def create_model():
    """
    Create EfficientNet-B0 with two output classes.
    """

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    return model


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_path):
    """
    Load a trained EfficientNet-B0 checkpoint.
    """

    model = create_model()

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# LOAD BOTH MODELS ONCE
# =========================================================

print("Loading Trustify AI models...")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Loading CIFAKE EfficientNet-B0...")
cifake_model = load_model(CIFAKE_MODEL_PATH)

print("Loading CASIA EfficientNet-B0...")
casia_model = load_model(CASIA_MODEL_PATH)

print("Both models loaded successfully.")


# =========================================================
# IMAGE PREPARATION
# =========================================================

def load_image(image_path):
    """
    Load and preprocess an image.
    """

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# =========================================================
# CIFAKE PREDICTION
# =========================================================

def predict_cifake(image_tensor):
    """
    CIFAKE mapping:
        FAKE = 0
        REAL = 1
    """

    with torch.no_grad():

        output = cifake_model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

    fake_score = probabilities[0][0].item()
    real_score = probabilities[0][1].item()

    return fake_score, real_score


# =========================================================
# CASIA PREDICTION
# =========================================================

def predict_casia(image_tensor):
    """
    CASIA mapping:
        EDITED = 0
        REAL = 1
    """

    with torch.no_grad():

        output = casia_model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

    edited_score = probabilities[0][0].item()
    real_score = probabilities[0][1].item()

    return edited_score, real_score


# =========================================================
# ANALYZE IMAGE
# =========================================================

def analyze_image(image_path):
    """
    Run both models and return the unified authenticity result.
    """

    image_tensor = load_image(image_path)

    # -----------------------------------------------------
    # CIFAKE
    # -----------------------------------------------------

    cifake_fake, cifake_real = predict_cifake(
        image_tensor
    )

    # -----------------------------------------------------
    # CASIA
    # -----------------------------------------------------

    casia_edited, casia_real = predict_casia(
        image_tensor
    )

    # -----------------------------------------------------
    # DECISION ENGINE
    # -----------------------------------------------------

    result = decide(
        cifake_fake_score=cifake_fake,
        casia_edited_score=casia_edited
    )

    # -----------------------------------------------------
    # RAW MODEL RESULTS
    # -----------------------------------------------------

    result["cifake_fake"] = round(
        cifake_fake * 100, 2
    )

    result["cifake_real"] = round(
        cifake_real * 100, 2
    )

    result["casia_edited"] = round(
        casia_edited * 100, 2
    )

    result["casia_real"] = round(
        casia_real * 100, 2
    )

    return result