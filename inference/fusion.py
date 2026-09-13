import sys
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIFD_DIR = PROJECT_ROOT / "ai_models" / "UniversalFakeDetect"
UNIFD_CHECKPOINT = UNIFD_DIR / "pretrained_weights" / "fc_weights.pth"

EFFICIENTNET_CHECKPOINT = (
    PROJECT_ROOT / "models" / "efficientnet_best.pth"
)

# Make UniFD's local models package importable
sys.path.insert(0, str(UNIFD_DIR))


# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0


# UniFD
from models import get_model


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMG_SIZE = 224


# ============================================================
# EFFICIENTNET PREPROCESSING
# ============================================================

efficientnet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# UNIFD PREPROCESSING
# ============================================================

UNIFD_MEAN = (
    0.48145466,
    0.4578275,
    0.40821073
)

UNIFD_STD = (
    0.26862954,
    0.26130258,
    0.27577711
)

unifd_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        UNIFD_MEAN,
        UNIFD_STD
    )
])


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

def load_efficientnet():

    print("\nLoading EfficientNet-B0...")

    model = efficientnet_b0(weights=None)

    # CIFAKE is binary classification:
    # 0 = FAKE
    # 1 = REAL

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    checkpoint = torch.load(
        EFFICIENTNET_CHECKPOINT,
        map_location=DEVICE
    )

    # efficientnet_best.pth is a raw state_dict
    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    print("EfficientNet loaded successfully!")

    return model


# ============================================================
# LOAD UNIFD
# ============================================================

def load_unifd():

    print("\nLoading UniFD...")

    model = get_model("CLIP:ViT-L/14")

    state_dict = torch.load(
        UNIFD_CHECKPOINT,
        map_location=DEVICE
    )

    model.fc.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("UniFD loaded successfully!")

    return model


# ============================================================
# LOAD BOTH MODELS ONCE
# ============================================================

efficientnet_model = load_efficientnet()
unifd_model = load_unifd()


# ============================================================
# EFFICIENTNET PREDICTION
# ============================================================

def get_efficientnet_score(image_path):

    """
    Returns:
        fake_probability
        real_probability
    """

    image = Image.open(image_path).convert("RGB")

    tensor = efficientnet_transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        logits = efficientnet_model(tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )

    # CIFAKE:
    # class 0 = FAKE
    # class 1 = REAL

    fake_probability = probabilities[0, 0].item()
    real_probability = probabilities[0, 1].item()

    return fake_probability, real_probability


# ============================================================
# UNIFD PREDICTION
# ============================================================

def get_unifd_score(image_path):

    """
    Returns UniFD sigmoid score.

    NOTE:
    This score is NOT assumed to be a calibrated
    fake probability.
    """

    image = Image.open(image_path).convert("RGB")

    tensor = unifd_transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        output = unifd_model(tensor)

        score = torch.sigmoid(output).item()

    return score


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def calibrate_unifd_score(
    unifd_score,
    low=0.0,
    high=1.0
):

    """
    Initial calibration layer.

    For the first experiment we keep UniFD's
    sigmoid score in [0,1].

    IMPORTANT:
    This is intentionally NOT treated as
    a true probability yet.

    Later, validation data can be used to
    replace this with learned calibration.
    """

    calibrated = (
        unifd_score - low
    ) / (high - low)

    calibrated = max(
        0.0,
        min(1.0, calibrated)
    )

    return calibrated


# ============================================================
# FUSION
# ============================================================

def fuse_scores(
    efficientnet_fake,
    unifd_score,
    efficientnet_weight=0.8,
    unifd_weight=0.2
):

    # Normalize weights
    total = efficientnet_weight + unifd_weight

    efficientnet_weight /= total
    unifd_weight /= total

    unifd_fake = calibrate_unifd_score(
        unifd_score
    )

    fused_fake_probability = (
        efficientnet_weight * efficientnet_fake
        +
        unifd_weight * unifd_fake
    )

    return fused_fake_probability


# ============================================================
# FINAL PREDICTION
# ============================================================

def predict(image_path):

    print("\n" + "=" * 60)
    print("DIGITAL MEDIA AUTHENTICITY ANALYSIS")
    print("=" * 60)

    print(f"Image: {image_path}")

    # --------------------------------------------------------
    # EfficientNet
    # --------------------------------------------------------

    efficientnet_fake, efficientnet_real = (
        get_efficientnet_score(image_path)
    )

    # --------------------------------------------------------
    # UniFD
    # --------------------------------------------------------

    unifd_score = get_unifd_score(image_path)

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    fused_fake = fuse_scores(
        efficientnet_fake,
        unifd_score,
        efficientnet_weight=0.8,
        unifd_weight=0.2
    )

    fused_real = 1.0 - fused_fake

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if fused_fake >= 0.5:

        label = "AI-GENERATED"
        confidence = fused_fake

    else:

        label = "REAL"
        confidence = fused_real

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nMODEL SCORES")
    print("-" * 60)

    print(
        f"EfficientNet Fake Score : "
        f"{efficientnet_fake:.4f}"
    )

    print(
        f"EfficientNet Real Score : "
        f"{efficientnet_real:.4f}"
    )

    print(
        f"UniFD Raw Score         : "
        f"{unifd_score:.4f}"
    )

    print(
        f"Fused Fake Score        : "
        f"{fused_fake:.4f}"
    )

    print(
        f"Fused Real Score        : "
        f"{fused_real:.4f}"
    )

    print("\nFINAL RESULT")
    print("-" * 60)

    print(f"Prediction : {label}")
    print(
        f"Confidence : {confidence * 100:.2f}%"
    )

    print("=" * 60)

    return {
        "label": label,
        "confidence": confidence,
        "efficientnet_fake": efficientnet_fake,
        "efficientnet_real": efficientnet_real,
        "unifd_score": unifd_score,
        "fused_fake": fused_fake,
        "fused_real": fused_real
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python inference\\fusion.py '
            '"path\\to\\image.jpg"'
        )

        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():

        print(
            f"\nERROR: Image not found:\n"
            f"{image_path}"
        )

        sys.exit(1)

    predict(image_path)