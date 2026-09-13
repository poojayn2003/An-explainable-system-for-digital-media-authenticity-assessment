import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = "datasets/CIFAKE/test"
MODEL_PATH = "models/efficientnet_best.pth"

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRUSTIFY AI - CIFAKE MODEL EVALUATION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print(f"\nLoading model: {MODEL_PATH}")


# ============================================================
# TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading CIFAKE test dataset...")

test_dataset = datasets.ImageFolder(
    root=TEST_PATH,
    transform=transform
)

print(f"Classes: {test_dataset.classes}")
print(f"Class mapping: {test_dataset.class_to_idx}")
print(f"Total test images: {len(test_dataset)}")


# ============================================================
# DATA LOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(f"Test batches: {len(test_loader)}")


# ============================================================
# MODEL
# ============================================================

model = efficientnet_b0(
    weights=None
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    2
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("✓ Trained model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING TEST EVALUATION")
print("=" * 70)

all_labels = []
all_predictions = []


with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(
        test_loader
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Processed "
                f"[{batch_idx + 1}/{len(test_loader)}] "
                f"batches"
            )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

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


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\n                 Predicted"
)

print(
    "              FAKE      REAL"
)

print(
    f"Actual FAKE   {cm[0][0]:6d}    {cm[0][1]:6d}"
)

print(
    f"Actual REAL   {cm[1][0]:6d}    {cm[1][1]:6d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
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

    print("=" * 70)
    print("GPU MEMORY")
    print("=" * 70)

    print(
        f"Allocated: {allocated:.2f} MB"
    )

    print(
        f"Reserved : {reserved:.2f} MB"
    )


print("\nEvaluation completed successfully.")