import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "datasets" / "CASIA2_clean"
VAL_DIR = DATA_DIR / "val"
MODEL_PATH = BASE_DIR / "models" / "casia_efficientnet_best.pth"


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("=" * 60)
print("CASIA THRESHOLD CALIBRATION")
print("=" * 60)

print(f"Device: {DEVICE}")

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

print()
print("Classes:", val_dataset.classes)
print("Class mapping:", val_dataset.class_to_idx)
print("Validation images:", len(val_dataset))


# ============================================================
# DATA LOADER
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading CASIA EfficientNet-B0...")

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support raw state_dict and checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()


# ============================================================
# COLLECT VALIDATION PREDICTIONS
# ============================================================

print()
print("Running validation inference...")

all_labels = []
all_edited_probabilities = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Class mapping:
        # EDITED = 0
        # REAL   = 1
        #
        # Therefore probability of EDITED = column 0

        edited_probability = probabilities[:, 0]

        all_edited_probabilities.extend(
            edited_probability.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# CONVERT LABELS
# ============================================================

all_labels = torch.tensor(all_labels).numpy()

# For metrics below:
#
# EDITED = 1
# REAL   = 0
#
# ImageFolder labels:
# EDITED = 0
# REAL   = 1

actual_edited = (all_labels == 0).astype(int)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

print()
print("=" * 100)
print("THRESHOLD ANALYSIS")
print("=" * 100)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'False Positive Rate':<20}"
)

print("-" * 100)


results = []


for threshold in thresholds:

    # If EDITED probability >= threshold:
    # prediction = EDITED (1)
    #
    # Otherwise:
    # prediction = REAL (0)

    predicted_edited = (
        torch.tensor(all_edited_probabilities) >= threshold
    ).numpy().astype(int)

    accuracy = accuracy_score(
        actual_edited,
        predicted_edited
    )

    precision = precision_score(
        actual_edited,
        predicted_edited,
        zero_division=0
    )

    recall = recall_score(
        actual_edited,
        predicted_edited,
        zero_division=0
    )

    f1 = f1_score(
        actual_edited,
        predicted_edited,
        zero_division=0
    )

    cm = confusion_matrix(
        actual_edited,
        predicted_edited,
        labels=[0, 1]
    )

    # Matrix:
    #
    #              Predicted
    #              REAL EDITED
    # Actual REAL
    #        EDITED
    #
    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": false_positive_rate
    })

    print(
        f"{threshold:<12.2f}"
        f"{accuracy * 100:<12.2f}"
        f"{precision * 100:<12.2f}"
        f"{recall * 100:<12.2f}"
        f"{f1 * 100:<12.2f}"
        f"{false_positive_rate * 100:<20.2f}"
    )


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_f1_result = max(
    results,
    key=lambda x: x["f1"]
)


# ============================================================
# LOW-FALSE-POSITIVE THRESHOLD
# ============================================================

# Select the best F1 threshold among thresholds
# where false positive rate <= 15%, if available.

low_fp_candidates = [
    r for r in results
    if r["fpr"] <= 0.15
]

if low_fp_candidates:

    best_low_fp = max(
        low_fp_candidates,
        key=lambda x: x["f1"]
    )

else:

    best_low_fp = min(
        results,
        key=lambda x: x["fpr"]
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("RECOMMENDED THRESHOLDS")
print("=" * 60)

print(
    f"Best F1 threshold: "
    f"{best_f1_result['threshold']:.2f}"
)

print(
    f"  Accuracy : "
    f"{best_f1_result['accuracy'] * 100:.2f}%"
)

print(
    f"  Precision: "
    f"{best_f1_result['precision'] * 100:.2f}%"
)

print(
    f"  Recall   : "
    f"{best_f1_result['recall'] * 100:.2f}%"
)

print(
    f"  F1       : "
    f"{best_f1_result['f1'] * 100:.2f}%"
)

print(
    f"  FPR      : "
    f"{best_f1_result['fpr'] * 100:.2f}%"
)

print()

print(
    f"Low-FP threshold: "
    f"{best_low_fp['threshold']:.2f}"
)

print(
    f"  Accuracy : "
    f"{best_low_fp['accuracy'] * 100:.2f}%"
)

print(
    f"  Precision: "
    f"{best_low_fp['precision'] * 100:.2f}%"
)

print(
    f"  Recall   : "
    f"{best_low_fp['recall'] * 100:.2f}%"
)

print(
    f"  F1       : "
    f"{best_low_fp['f1'] * 100:.2f}%"
)

print(
    f"  FPR      : "
    f"{best_low_fp['fpr'] * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX FOR BEST F1
# ============================================================

threshold = best_f1_result["threshold"]

predicted_edited = (
    torch.tensor(all_edited_probabilities) >= threshold
).numpy().astype(int)

cm = confusion_matrix(
    actual_edited,
    predicted_edited,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print()
print("=" * 60)
print("CONFUSION MATRIX - BEST F1 THRESHOLD")
print("=" * 60)

print()
print("                 Predicted")
print("              REAL    EDITED")
print(
    f"Actual REAL   {tn:4d}    {fp:4d}"
)
print(
    f"Actual EDITED {fn:4d}    {tp:4d}"
)

print()
print("Calibration complete.")