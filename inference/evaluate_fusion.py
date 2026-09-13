import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CIFAKE_TRAIN = BASE_DIR / "datasets" / "CIFAKE" / "train"
EFFICIENTNET_PATH = BASE_DIR / "models" / "efficientnet_best.pth"

UNIFD_DIR = BASE_DIR / "ai_models" / "UniversalFakeDetect"
UNIFD_CHECKPOINT = UNIFD_DIR / "pretrained_weights" / "fc_weights.pth"

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224

# 1000 REAL + 1000 FAKE
SAMPLES_PER_CLASS = 1000

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUSTIFY AI - EFFICIENTNET + UNIFD FUSION EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# CHECK FILES
# ============================================================

if not EFFICIENTNET_PATH.exists():
    raise FileNotFoundError(
        f"EfficientNet model not found:\n{EFFICIENTNET_PATH}"
    )

if not UNIFD_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"UniFD checkpoint not found:\n{UNIFD_CHECKPOINT}"
    )

# ============================================================
# EFFICIENTNET TRANSFORM
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
# UNIFD TRANSFORM
# ============================================================

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

efficientnet = efficientnet_b0(weights=None)

in_features = efficientnet.classifier[1].in_features

efficientnet.classifier[1] = nn.Linear(
    in_features,
    2
)

state_dict = torch.load(
    EFFICIENTNET_PATH,
    map_location=DEVICE,
    weights_only=True
)

efficientnet.load_state_dict(state_dict)

efficientnet = efficientnet.to(DEVICE)
efficientnet.eval()

print("✓ EfficientNet loaded")

# ============================================================
# LOAD UNIFD
# ============================================================

print("\nLoading UniFD...")

# Make sure the UniFD directory is searched first.
sys.path.insert(0, str(UNIFD_DIR))

from models import get_model

MODEL_NAME = "CLIP:ViT-L/14"

unifd = get_model(MODEL_NAME)

unifd_state = torch.load(
    UNIFD_CHECKPOINT,
    map_location=DEVICE
)

unifd.fc.load_state_dict(unifd_state)

unifd = unifd.to(DEVICE)
unifd.eval()

print("✓ UniFD loaded")

# ============================================================
# COLLECT CIFAKE IMAGES
# ============================================================

print("\nCollecting CIFAKE validation images...")

fake_dir = CIFAKE_TRAIN / "FAKE"
real_dir = CIFAKE_TRAIN / "REAL"

if not fake_dir.exists():
    raise FileNotFoundError(f"Missing: {fake_dir}")

if not real_dir.exists():
    raise FileNotFoundError(f"Missing: {real_dir}")

fake_images = [
    p for p in fake_dir.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
]

real_images = [
    p for p in real_dir.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
]

random.shuffle(fake_images)
random.shuffle(real_images)

fake_images = fake_images[:SAMPLES_PER_CLASS]
real_images = real_images[:SAMPLES_PER_CLASS]

image_paths = []

# FAKE = 1
for path in fake_images:
    image_paths.append((path, 1))

# REAL = 0
for path in real_images:
    image_paths.append((path, 0))

random.shuffle(image_paths)

print(f"FAKE validation images: {len(fake_images)}")
print(f"REAL validation images: {len(real_images)}")
print(f"Total validation images: {len(image_paths)}")

# ============================================================
# INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION INFERENCE")
print("=" * 70)

labels = []

efficientnet_scores = []
unifd_logits = []

with torch.inference_mode():

    for index, (image_path, label) in enumerate(image_paths):

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # EfficientNet
        # ----------------------------------------------------

        eff_image = efficientnet_transform(image)
        eff_image = eff_image.unsqueeze(0).to(DEVICE)

        eff_output = efficientnet(eff_image)

        eff_probability = torch.softmax(
            eff_output,
            dim=1
        )[0, 0].item()

        # Class 0 = FAKE
        efficientnet_scores.append(
            eff_probability
        )

        # ----------------------------------------------------
        # UniFD
        # ----------------------------------------------------

        uni_image = unifd_transform(image)
        uni_image = uni_image.unsqueeze(0).to(DEVICE)

        uni_output = unifd(uni_image)

        uni_logit = uni_output.item()

        unifd_logits.append(
            uni_logit
        )

        labels.append(label)

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(image_paths)}"
            )

# ============================================================
# NUMPY
# ============================================================

labels = np.array(labels)

efficientnet_scores = np.array(
    efficientnet_scores
)

unifd_logits = np.array(
    unifd_logits
)

# ============================================================
# EFFICIENTNET BASELINE
# ============================================================

efficientnet_predictions = (
    efficientnet_scores >= 0.5
).astype(int)

eff_accuracy = accuracy_score(
    labels,
    efficientnet_predictions
)

eff_precision = precision_score(
    labels,
    efficientnet_predictions,
    zero_division=0
)

eff_recall = recall_score(
    labels,
    efficientnet_predictions,
    zero_division=0
)

eff_f1 = f1_score(
    labels,
    efficientnet_predictions,
    zero_division=0
)

# ============================================================
# UNIFD CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATING UNIFD")
print("=" * 70)

# Convert UniFD raw logits into a learned
# probability of FAKE.

calibrator = LogisticRegression(
    random_state=SEED
)

calibrator.fit(
    unifd_logits.reshape(-1, 1),
    labels
)

unifd_fake_probability = calibrator.predict_proba(
    unifd_logits.reshape(-1, 1)
)[:, 1]

unifd_predictions = (
    unifd_fake_probability >= 0.5
).astype(int)

uni_accuracy = accuracy_score(
    labels,
    unifd_predictions
)

uni_precision = precision_score(
    labels,
    unifd_predictions,
    zero_division=0
)

uni_recall = recall_score(
    labels,
    unifd_predictions,
    zero_division=0
)

uni_f1 = f1_score(
    labels,
    unifd_predictions,
    zero_division=0
)

# ============================================================
# FUSION SEARCH
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR BEST FUSION WEIGHT")
print("=" * 70)

best_weight = None
best_accuracy = -1
best_predictions = None

for efficientnet_weight in np.arange(
    0.0,
    1.01,
    0.05
):

    unifd_weight = 1.0 - efficientnet_weight

    fused_probability = (
        efficientnet_weight * efficientnet_scores
        +
        unifd_weight * unifd_fake_probability
    )

    predictions = (
        fused_probability >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    print(
        f"EfficientNet={efficientnet_weight:.2f} | "
        f"UniFD={unifd_weight:.2f} | "
        f"Accuracy={accuracy:.4f}"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_weight = efficientnet_weight

        best_predictions = predictions.copy()

# ============================================================
# BEST FUSION METRICS
# ============================================================

best_unifd_weight = 1.0 - best_weight

fusion_precision = precision_score(
    labels,
    best_predictions,
    zero_division=0
)

fusion_recall = recall_score(
    labels,
    best_predictions,
    zero_division=0
)

fusion_f1 = f1_score(
    labels,
    best_predictions,
    zero_division=0
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print("\nEfficientNet-B0")
print(
    f"Accuracy : {eff_accuracy:.4f}"
)
print(
    f"Precision: {eff_precision:.4f}"
)
print(
    f"Recall   : {eff_recall:.4f}"
)
print(
    f"F1 Score : {eff_f1:.4f}"
)

print("\nUniFD - Calibrated")
print(
    f"Accuracy : {uni_accuracy:.4f}"
)
print(
    f"Precision: {uni_precision:.4f}"
)
print(
    f"Recall   : {uni_recall:.4f}"
)
print(
    f"F1 Score : {uni_f1:.4f}"
)

print("\nBest Fusion")
print(
    f"EfficientNet weight: "
    f"{best_weight:.2f}"
)

print(
    f"UniFD weight: "
    f"{best_unifd_weight:.2f}"
)

print(
    f"Accuracy : {best_accuracy:.4f}"
)

print(
    f"Precision: {fusion_precision:.4f}"
)

print(
    f"Recall   : {fusion_recall:.4f}"
)

print(
    f"F1 Score : {fusion_f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("BEST FUSION CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        labels,
        best_predictions,
        target_names=["REAL", "FAKE"],
        digits=4,
        zero_division=0
    )
)

# ============================================================
# SAVE CALIBRATION VALUES
# ============================================================

calibration_path = (
    BASE_DIR
    / "models"
    / "unifd_calibration.npz"
)

np.savez(
    calibration_path,
    coefficient=calibrator.coef_,
    intercept=calibrator.intercept_,
    efficientnet_weight=best_weight,
    unifd_weight=best_unifd_weight
)

print(
    f"\n✓ Calibration saved to:"
    f"\n  {calibration_path}"
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
        f"Allocated: {allocated:.2f} MB"
    )

    print(
        f"Reserved : {reserved:.2f} MB"
    )

print("\nValidation evaluation completed.")