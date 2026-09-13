"""
Trustify AI - Unified Image Authenticity Predictor

Models:
    CIFAKE EfficientNet-B0
    CASIA EfficientNet-B0

Input:
    One image path

Output:
    CIFAKE score
    CASIA score
    Final decision
    Confidence
    Trust score
"""

import sys
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

CIFAKE_MODEL_PATH = BASE_DIR / "models" / "efficientnet_best.pth"
CASIA_MODEL_PATH = BASE_DIR / "models" / "casia_efficientnet_best.pth"


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TRUSTIFY AI - IMAGE AUTHENTICITY ANALYSIS")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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
# LOAD IMAGE
# =========================================================

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# =========================================================
# CIFAKE PREDICTION
# =========================================================

def predict_cifake(model, image_tensor):

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

    # CIFAKE mapping:
    # FAKE = 0
    # REAL = 1

    fake_score = probabilities[0][0].item()
    real_score = probabilities[0][1].item()

    return fake_score, real_score


# =========================================================
# CASIA PREDICTION
# =========================================================

def predict_casia(model, image_tensor):

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

    # CASIA mapping:
    # EDITED = 0
    # REAL = 1

    edited_score = probabilities[0][0].item()
    real_score = probabilities[0][1].item()

    return edited_score, real_score


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_image(image_path):

    print("\nLoading models...")

    print("Loading CIFAKE EfficientNet-B0...")
    cifake_model = load_model(CIFAKE_MODEL_PATH)

    print("Loading CASIA EfficientNet-B0...")
    casia_model = load_model(CASIA_MODEL_PATH)

    print("Models loaded successfully.")

    print("\nAnalyzing image...")

    image_tensor = load_image(image_path)

    # -----------------------------------------------------
    # CIFAKE
    # -----------------------------------------------------

    cifake_fake, cifake_real = predict_cifake(
        cifake_model,
        image_tensor
    )

    # -----------------------------------------------------
    # CASIA
    # -----------------------------------------------------

    casia_edited, casia_real = predict_casia(
        casia_model,
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
    # ADD RAW MODEL RESULTS
    # -----------------------------------------------------

    result["cifake_fake"] = round(cifake_fake * 100, 2)
    result["cifake_real"] = round(cifake_real * 100, 2)

    result["casia_edited"] = round(casia_edited * 100, 2)
    result["casia_real"] = round(casia_real * 100, 2)

    result["image"] = str(image_path)

    return result


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("\nUsage:")
        print(
            r'python inference\predict.py "path\to\image.jpg"'
        )

        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():

        print(f"\nERROR: Image not found:")
        print(image_path)

        sys.exit(1)

    result = analyze_image(image_path)

    print("\n")
    print("=" * 70)
    print("FINAL AUTHENTICITY RESULT")
    print("=" * 70)

    print(f"\nImage: {result['image']}")

    print("\nCIFAKE DETECTOR")
    print(f"  AI-Generated / FAKE : {result['cifake_fake']}%")
    print(f"  REAL                : {result['cifake_real']}%")

    print("\nCASIA MANIPULATION DETECTOR")
    print(f"  EDITED              : {result['casia_edited']}%")
    print(f"  REAL                : {result['casia_real']}%")

    print("\nDECISION ENGINE")
    print(f"  Prediction          : {result['prediction']}")
    print(f"  Confidence          : {result['confidence']}%")
    print(f"  Trust Score         : {result['trust_score']}/100")
    print(f"  Reason              : {result['reason']}")

    print("\n" + "=" * 70)

