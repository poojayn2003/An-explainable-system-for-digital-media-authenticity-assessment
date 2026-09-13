import os
import sys
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image

from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = BASE_DIR / "datasets" / "CIFAKE" / "test"

EFFICIENTNET_PATH = (
    BASE_DIR / "models" / "efficientnet_best.pth"
)

UNIFD_DIR = (
    BASE_DIR / "ai_models" / "UniversalFakeDetect"
)

UNIFD_CHECKPOINT = (
    UNIFD_DIR / "pretrained_weights" / "fc_weights.pth"
)

CALIBRATION_PATH = (
    BASE_DIR / "models" / "unifd_calibration.npz"
)


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUSTIFY AI - FINAL CIFAKE FUSION TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    TEST_PATH,
    EFFICIENTNET_PATH,
    UNIFD_CHECKPOINT,
    CALIBRATION_PATH
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# LOAD CALIBRATION
# ============================================================

print("\nLoading UniFD calibration...")

calibration = np.load(
    CALIBRATION_PATH
)

coefficient = calibration["coefficient"]
intercept = calibration["intercept"]

efficientnet_weight = float(
    calibration["efficientnet_weight"]
)

unifd_weight = float(
    calibration["unifd_weight"]
)

print(
    f"EfficientNet weight: "
    f"{efficientnet_weight:.2f}"
)

print(
    f"UniFD weight: "
    f"{unifd_weight:.2f}"
)


# ============================================================
# TRANSFORMS
# ============================================================

efficientnet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


unifd_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

print("\nLoading EfficientNet-B0...")

efficientnet = efficientnet_b0(
    weights=None
)

in_features = (
    efficientnet.classifier[1].in_features
)

efficientnet.classifier[1] = nn.Linear(
    in_features,
    2
)

state_dict = torch.load(
    EFFICIENTNET_PATH,
    map_location=DEVICE,
    weights_only=True
)

efficientnet.load_state_dict(
    state_dict
)

efficientnet = efficientnet.to(DEVICE)

efficientnet.eval()

print("✓ EfficientNet loaded")


# ============================================================
# LOAD UNIFD
# ============================================================

print("\nLoading UniFD...")

sys.path.insert(
    0,
    str(UNIFD_DIR)
)

from models import get_model

unifd = get_model(
    "CLIP:ViT-L/14"
)

unifd_state = torch.load(
    UNIFD_CHECKPOINT,
    map_location=DEVICE
)

unifd.fc.load_state_dict(
    unifd_state
)

unifd = unifd.to(DEVICE)

unifd.eval()

print("✓ UniFD loaded")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading CIFAKE test dataset...")

test_dataset = datasets.ImageFolder(
    root=str(TEST_PATH)
)

print(
    f"Classes: {test_dataset.classes}"
)

print(
    f"Class mapping: "
    f"{test_dataset.class_to_idx}"
)

print(
    f"Total test images: "
    f"{len(test_dataset)}"
)


# ============================================================
# RESULTS STORAGE
# ============================================================

labels = []

efficientnet_scores = []

unifd_logits = []


# ============================================================
# TEST INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FINAL TEST")
print("=" * 70)

with torch.inference_mode():

    for index, (image_path, label) in enumerate(
        test_dataset.samples
    ):

        image = Image.open(
            image_path
        ).convert("RGB")


        # ====================================================
        # EFFICIENTNET
        # ====================================================

        eff_image = efficientnet_transform(
            image
        )

        eff_image = eff_image.unsqueeze(
            0
        ).to(DEVICE)

        eff_output = efficientnet(
            eff_image
        )

        eff_fake_probability = torch.softmax(
            eff_output,
            dim=1
        )[0, 0].item()

        efficientnet_scores.append(
            eff_fake_probability
        )


        # ====================================================
        # UNIFD
        # ====================================================

        uni_image = unifd_transform(
            image
        )

        uni_image = uni_image.unsqueeze(
            0
        ).to(DEVICE)

        uni_output = unifd(
            uni_image
        )

        uni_logit = uni_output.item()

        unifd_logits.append(
            uni_logit
        )


        labels.append(
            label
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(test_dataset)}"
            )


# ============================================================
# NUMPY
# ============================================================

labels = np.array(
    labels
)

efficientnet_scores = np.array(
    efficientnet_scores
)

unifd_logits = np.array(
    unifd_logits
)


# ============================================================
# UNI FD CALIBRATION
# ============================================================

# Logistic regression:
#
# P(FAKE) = sigmoid(
#     coefficient * UniFD_logit
#     + intercept
# )

unifd_fake_probability = 1 / (
    1 + np.exp(
        -(
            coefficient[0] * unifd_logits
            + intercept[0]
        )
    )
)


# ============================================================
# PREDICTIONS
# ============================================================

# ============================================================
# PREDICTIONS
# ============================================================
#
# CIFAKE ImageFolder mapping:
#
# FAKE = 0
# REAL = 1
#
# Our model scores represent:
#
# fake_probability >= 0.5  -> FAKE -> class 0
# fake_probability <  0.5  -> REAL -> class 1
#


efficientnet_predictions = np.where(
    efficientnet_scores >= 0.5,
    0,   # FAKE
    1    # REAL
)


unifd_predictions = np.where(
    unifd_fake_probability >= 0.5,
    0,   # FAKE
    1    # REAL
)


# ============================================================
# FUSION
# ============================================================

fused_probability = (
    efficientnet_weight
    * efficientnet_scores
    +
    unifd_weight
    * unifd_fake_probability
)


fusion_predictions = np.where(
    fused_probability >= 0.5,
    0,   # FAKE
    1    # REAL
)

# ============================================================
# METRIC FUNCTION
# ============================================================

def print_metrics(
    name,
    predictions
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    print(
        f"\n{name}"
    )

    print(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    return accuracy


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CIFAKE TEST RESULTS")
print("=" * 70)


eff_accuracy = print_metrics(
    "EfficientNet-B0",
    efficientnet_predictions
)


uni_accuracy = print_metrics(
    "UniFD - Calibrated",
    unifd_predictions
)


fusion_accuracy = print_metrics(
    "EfficientNet + UniFD Fusion",
    fusion_predictions
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"EfficientNet : "
    f"{eff_accuracy * 100:.2f}%"
)

print(
    f"UniFD        : "
    f"{uni_accuracy * 100:.2f}%"
)

print(
    f"Fusion       : "
    f"{fusion_accuracy * 100:.2f}%"
)

print(
    "\nFusion improvement over EfficientNet: "
    f"{(fusion_accuracy - eff_accuracy) * 100:+.2f} "
    "percentage points"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    labels,
    fusion_predictions
)

print("\n" + "=" * 70)
print("FUSION CONFUSION MATRIX")
print("=" * 70)

print(
    "\n                 Predicted"
)

print(
    "              FAKE      REAL"
)

print(
    f"Actual FAKE   "
    f"{cm[0][0]:6d}    "
    f"{cm[0][1]:6d}"
)

print(
    f"Actual REAL   "
    f"{cm[1][0]:6d}    "
    f"{cm[1][1]:6d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("FUSION CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        labels,
        fusion_predictions,
        target_names=[
            "FAKE",
            "REAL"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    print("\n" + "=" * 70)
    print("GPU MEMORY")
    print("=" * 70)

    print(
        f"Allocated: "
        f"{allocated:.2f} MB"
    )

    print(
        f"Reserved : "
        f"{reserved:.2f} MB"
    )


print("\nFinal fusion test completed successfully.")