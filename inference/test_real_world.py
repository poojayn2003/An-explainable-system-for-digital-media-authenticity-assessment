from pathlib import Path
import sys

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# PATHS
# ============================================================

TEST_DIR = BASE_DIR / "datasets" / "real_world_test" / "REAL"

CIFAKE_MODEL_PATH = BASE_DIR / "models" / "efficientnet_best.pth"
CASIA_MODEL_PATH = BASE_DIR / "models" / "casia_efficientnet_best.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 100)
print("TRUSTIFY AI - REAL-WORLD GENERALIZATION TEST")
print("=" * 100)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

imagenet_weights = EfficientNet_B0_Weights.DEFAULT

cifake_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_weights.transforms().mean,
        std=imagenet_weights.transforms().std
    )
])

casia_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CIFAKE MODEL
# ============================================================

print("\nLoading CIFAKE EfficientNet-B0...")

cifake_model = efficientnet_b0(weights=None)

in_features = cifake_model.classifier[1].in_features

cifake_model.classifier[1] = nn.Linear(
    in_features,
    2
)

cifake_state = torch.load(
    CIFAKE_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

cifake_model.load_state_dict(cifake_state)

cifake_model = cifake_model.to(DEVICE)
cifake_model.eval()

print("CIFAKE model loaded successfully.")


# ============================================================
# LOAD CASIA MODEL
# ============================================================

print("\nLoading CASIA EfficientNet-B0...")

casia_model = models.efficientnet_b0(weights=None)

in_features = casia_model.classifier[1].in_features

casia_model.classifier[1] = nn.Linear(
    in_features,
    2
)

casia_state = torch.load(
    CASIA_MODEL_PATH,
    map_location=DEVICE
)

casia_model.load_state_dict(casia_state)

casia_model = casia_model.to(DEVICE)
casia_model.eval()

print("CASIA model loaded successfully.")


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_cifake(image):
    """
    CIFAKE:
    class 0 = AI-GENERATED
    class 1 = REAL
    """

    tensor = cifake_transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = cifake_model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    fake_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    if fake_probability >= real_probability:
        prediction = "AI-GENERATED"
        confidence = fake_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    return (
        prediction,
        confidence,
        fake_probability,
        real_probability
    )


def predict_casia(image):
    """
    CASIA:
    class 0 = EDITED
    class 1 = REAL
    """

    tensor = casia_transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = casia_model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    edited_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    if edited_probability >= real_probability:
        prediction = "AI-EDITED/MANIPULATED"
        confidence = edited_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    return (
        prediction,
        confidence,
        edited_probability,
        real_probability
    )


# ============================================================
# FIND IMAGES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

images = sorted(
    [
        p for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

print("\n" + "=" * 100)
print("TEST SET")
print("=" * 100)
print(f"Folder : {TEST_DIR}")
print(f"Images : {len(images)}")


# ============================================================
# RUN TEST
# ============================================================

results = []

for index, image_path in enumerate(images, 1):

    print(
        f"\n[{index}/{len(images)}] "
        f"{image_path.name}"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    # CIFAKE
    (
        cifake_prediction,
        cifake_confidence,
        fake_probability,
        cifake_real_probability
    ) = predict_cifake(image)

    # CASIA
    (
        casia_prediction,
        casia_confidence,
        edited_probability,
        casia_real_probability
    ) = predict_casia(image)

    print(
        f"  CIFAKE : {cifake_prediction} "
        f"({cifake_confidence * 100:.2f}%)"
    )

    print(
        f"           FAKE={fake_probability * 100:.2f}% "
        f"REAL={cifake_real_probability * 100:.2f}%"
    )

    print(
        f"  CASIA  : {casia_prediction} "
        f"({casia_confidence * 100:.2f}%)"
    )

    print(
        f"           EDITED={edited_probability * 100:.2f}% "
        f"REAL={casia_real_probability * 100:.2f}%"
    )

    results.append({
        "image": image_path.name,
        "cifake_prediction": cifake_prediction,
        "fake_probability": fake_probability,
        "cifake_real_probability": cifake_real_probability,
        "casia_prediction": casia_prediction,
        "edited_probability": edited_probability,
        "casia_real_probability": casia_real_probability
    })


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n\n")
print("=" * 130)
print("DETAILED RESULTS")
print("=" * 130)

print(
    f"{'IMAGE':28} "
    f"{'CIFAKE':16} "
    f"{'FAKE %':9} "
    f"{'REAL %':9} "
    f"{'CASIA':24} "
    f"{'EDITED %':10} "
    f"{'REAL %':9}"
)

print("-" * 130)

for r in results:

    print(
        f"{r['image'][:28]:28} "
        f"{r['cifake_prediction'][:16]:16} "
        f"{r['fake_probability'] * 100:8.2f} "
        f"{r['cifake_real_probability'] * 100:8.2f} "
        f"{r['casia_prediction'][:24]:24} "
        f"{r['edited_probability'] * 100:9.2f} "
        f"{r['casia_real_probability'] * 100:8.2f}"
    )


# ============================================================
# EXTERNAL REAL-WORLD METRICS
# ============================================================

total = len(results)

cifake_real_count = sum(
    1
    for r in results
    if r["cifake_prediction"] == "REAL"
)

casia_real_count = sum(
    1
    for r in results
    if r["casia_prediction"] == "REAL"
)

cifake_false_positive_rate = (
    (total - cifake_real_count) / total * 100
)

casia_false_positive_rate = (
    (total - casia_real_count) / total * 100
)


# ============================================================
# AGREEMENT
# ============================================================

both_real = sum(
    1
    for r in results
    if (
        r["cifake_prediction"] == "REAL"
        and
        r["casia_prediction"] == "REAL"
    )
)

cifake_fake_casia_real = sum(
    1
    for r in results
    if (
        r["cifake_prediction"] == "AI-GENERATED"
        and
        r["casia_prediction"] == "REAL"
    )
)

cifake_real_casia_edited = sum(
    1
    for r in results
    if (
        r["cifake_prediction"] == "REAL"
        and
        r["casia_prediction"] == "AI-EDITED/MANIPULATED"
    )
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("EXTERNAL REAL-WORLD EVALUATION")
print("=" * 80)

print(
    f"Total genuine images        : {total}"
)

print()

print(
    f"CIFAKE REAL predictions     : "
    f"{cifake_real_count}/{total}"
)

print(
    f"CIFAKE false positives      : "
    f"{total - cifake_real_count}/{total}"
)

print(
    f"CIFAKE external FPR         : "
    f"{cifake_false_positive_rate:.2f}%"
)

print()

print(
    f"CASIA REAL predictions      : "
    f"{casia_real_count}/{total}"
)

print(
    f"CASIA false positives       : "
    f"{total - casia_real_count}/{total}"
)

print(
    f"CASIA external FPR         : "
    f"{casia_false_positive_rate:.2f}%"
)

print()

print(
    f"Both models predicted REAL  : "
    f"{both_real}/{total}"
)

print(
    f"CIFAKE=AI, CASIA=REAL      : "
    f"{cifake_fake_casia_real}/{total}"
)

print(
    f"CIFAKE=REAL, CASIA=EDITED  : "
    f"{cifake_real_casia_edited}/{total}"
)

print("=" * 80)