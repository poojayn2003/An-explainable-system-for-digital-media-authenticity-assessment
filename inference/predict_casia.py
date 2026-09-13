from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "casia_efficientnet_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# Must match validation/test preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading CASIA EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    2
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print(f"Device: {DEVICE}")


# ============================================================
# PREDICTION
# ============================================================

def predict(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    edited_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    if edited_probability >= real_probability:

        label = "AI-EDITED / MANIPULATED"
        confidence = edited_probability

    else:

        label = "REAL"
        confidence = real_probability

    return label, confidence, edited_probability, real_probability


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            'python inference\\predict_casia.py "path\\to\\image.jpg"'
        )
        sys.exit(1)

    image_path = sys.argv[1]

    try:

        label, confidence, edited_prob, real_prob = predict(
            image_path
        )

        print("\n" + "=" * 50)
        print("CASIA IMAGE AUTHENTICITY RESULT")
        print("=" * 50)

        print(f"Image: {image_path}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence * 100:.2f}%")

        print("\nClass probabilities:")
        print(
            f"EDITED: {edited_prob * 100:.2f}%"
        )
        print(
            f"REAL:   {real_prob * 100:.2f}%"
        )

        print("=" * 50)

    except Exception as e:

        print(f"\nError: {e}")