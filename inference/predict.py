"""
Trustify AI - Unified Image Authenticity Predictor

Models:
    CIFAKE EfficientNet-B0
    CASIA EfficientNet-B0

Pipeline:
    Image
      ↓
    CIFAKE prediction
      ↓
    CASIA prediction
      ↓
    Decision Engine
      ↓
    Grad-CAM Explainability
      ↓
    Final Result
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

from inference.decision_engine import decide
from inference.gradcam import GradCAM, save_heatmap


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CIFAKE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "efficientnet_best.pth"
)

CASIA_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "casia_efficientnet_best.pth"
)

GRADCAM_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "gradcam"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRUSTIFY AI - IMAGE AUTHENTICITY ANALYSIS")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# MODEL CREATION
# =========================================================

def create_model():
    """
    Create EfficientNet-B0 with two output classes.
    """

    model = models.efficientnet_b0(
        weights=None
    )

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

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    return model


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    return image, tensor.to(DEVICE)


# =========================================================
# CIFAKE PREDICTION
# =========================================================

def predict_cifake(
    model,
    image_tensor
):

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    # CIFAKE mapping:
    # FAKE = 0
    # REAL = 1

    fake_score = probabilities[
        0,
        0
    ].item()

    real_score = probabilities[
        0,
        1
    ].item()

    return fake_score, real_score


# =========================================================
# CASIA PREDICTION
# =========================================================

def predict_casia(
    model,
    image_tensor
):

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    # CASIA mapping:
    # EDITED = 0
    # REAL = 1

    edited_score = probabilities[
        0,
        0
    ].item()

    real_score = probabilities[
        0,
        1
    ].item()

    return edited_score, real_score


# =========================================================
# GENERATE GRAD-CAM
# =========================================================

def generate_gradcam(
    model,
    image_tensor,
    original_image,
    target_class,
    output_path
):

    # EfficientNet-B0 final convolutional feature layer
    target_layer = model.features[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    try:

        cam = gradcam.generate(
            image_tensor,
            target_class
        )

        save_heatmap(
            original_image,
            cam,
            str(output_path)
        )

    finally:

        gradcam.remove_hooks()

    return output_path


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_image(image_path):

    print("\nLoading models...")

    # -----------------------------------------------------
    # CIFAKE
    # -----------------------------------------------------

    print(
        "Loading CIFAKE EfficientNet-B0..."
    )

    cifake_model = load_model(
        CIFAKE_MODEL_PATH
    )

    # -----------------------------------------------------
    # CASIA
    # -----------------------------------------------------

    print(
        "Loading CASIA EfficientNet-B0..."
    )

    casia_model = load_model(
        CASIA_MODEL_PATH
    )

    print(
        "Models loaded successfully."
    )

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    print(
        "\nAnalyzing image..."
    )

    original_image, image_tensor = load_image(
        image_path
    )

    # =====================================================
    # CIFAKE
    # =====================================================

    cifake_fake, cifake_real = predict_cifake(
        cifake_model,
        image_tensor
    )

    # =====================================================
    # CASIA
    # =====================================================

    casia_edited, casia_real = predict_casia(
        casia_model,
        image_tensor
    )

    # =====================================================
    # DECISION ENGINE
    # =====================================================

    result = decide(
        cifake_fake_score=cifake_fake,
        casia_edited_score=casia_edited
    )

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    GRADCAM_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = Path(
        image_path
    ).stem

    # =====================================================
    # CIFAKE GRAD-CAM
    # =====================================================

    print(
        "\nGenerating CIFAKE Grad-CAM..."
    )

    # CIFAKE:
    # FAKE = 0
    # REAL = 1

    cifake_target_class = 0

    cifake_heatmap_path = (
        GRADCAM_OUTPUT_DIR
        / f"{filename}_cifake_gradcam.jpg"
    )

    generate_gradcam(
        cifake_model,
        image_tensor,
        original_image,
        cifake_target_class,
        cifake_heatmap_path
    )

    print(
        f"✓ CIFAKE heatmap saved:"
        f"\n  {cifake_heatmap_path}"
    )

    # =====================================================
    # CASIA GRAD-CAM
    # =====================================================

    print(
        "\nGenerating CASIA Grad-CAM..."
    )

    # CASIA:
    # EDITED = 0
    # REAL = 1

    casia_target_class = 0

    casia_heatmap_path = (
        GRADCAM_OUTPUT_DIR
        / f"{filename}_casia_gradcam.jpg"
    )

    generate_gradcam(
        casia_model,
        image_tensor,
        original_image,
        casia_target_class,
        casia_heatmap_path
    )

    print(
        f"✓ CASIA heatmap saved:"
        f"\n  {casia_heatmap_path}"
    )

    # =====================================================
    # ADD RAW MODEL RESULTS
    # =====================================================

    result["cifake_fake"] = round(
        cifake_fake * 100,
        2
    )

    result["cifake_real"] = round(
        cifake_real * 100,
        2
    )

    result["casia_edited"] = round(
        casia_edited * 100,
        2
    )

    result["casia_real"] = round(
        casia_real * 100,
        2
    )

    # =====================================================
    # IMAGE
    # =====================================================

    result["image"] = str(
        image_path
    )

    # =====================================================
    # HEATMAP PATHS
    # =====================================================

    result["cifake_heatmap"] = str(
        cifake_heatmap_path
    )

    result["casia_heatmap"] = str(
        casia_heatmap_path
    )

    # =====================================================
    # RETURN
    # =====================================================

    return result


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python inference\\predict.py '
            '"path\\to\\image.jpg"'
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            "\nERROR: Image not found:"
        )

        print(
            image_path
        )

        sys.exit(1)

    result = analyze_image(
        image_path
    )

    print("\n")

    print("=" * 70)

    print(
        "FINAL AUTHENTICITY RESULT"
    )

    print("=" * 70)

    print(
        f"\nImage: {result['image']}"
    )

    # -----------------------------------------------------
    # CIFAKE
    # -----------------------------------------------------

    print(
        "\nCIFAKE DETECTOR"
    )

    print(
        f"  AI-Generated / FAKE : "
        f"{result['cifake_fake']}%"
    )

    print(
        f"  REAL                : "
        f"{result['cifake_real']}%"
    )

    # -----------------------------------------------------
    # CASIA
    # -----------------------------------------------------

    print(
        "\nCASIA MANIPULATION DETECTOR"
    )

    print(
        f"  EDITED              : "
        f"{result['casia_edited']}%"
    )

    print(
        f"  REAL                : "
        f"{result['casia_real']}%"
    )

    # -----------------------------------------------------
    # DECISION
    # -----------------------------------------------------

    print(
        "\nDECISION ENGINE"
    )

    print(
        f"  Prediction          : "
        f"{result['prediction']}"
    )

    print(
        f"  Confidence          : "
        f"{result['confidence']}%"
    )

    print(
        f"  Trust Score         : "
        f"{result['trust_score']}/100"
    )

    print(
        f"  Reason              : "
        f"{result['reason']}"
    )

    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    print(
        "\nEXPLAINABILITY"
    )

    print(
        f"  Explanation:"
    )

    print(
        f"  {result['explanation']}"
    )

    # -----------------------------------------------------
    # EVIDENCE
    # -----------------------------------------------------

    print(
        "\n  Evidence:"
    )

    for evidence in result["evidence"]:

        print(
            f"  - {evidence}"
        )

    # -----------------------------------------------------
    # HEATMAP
    # -----------------------------------------------------

    print(
        "\nGRAD-CAM HEATMAPS"
    )

    print(
        f"  CIFAKE:"
    )

    print(
        f"  {result['cifake_heatmap']}"
    )

    print(
        f"\n  CASIA:"
    )

    print(
        f"  {result['casia_heatmap']}"
    )

    print(
        "\n" + "=" * 70
    )