import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import cv2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "casia_efficientnet_best.pth"
OUTPUT_DIR = BASE_DIR / "outputs" / "gradcam_casia"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading CASIA EfficientNet-B0...")
print(f"Device: {DEVICE}")


# ============================================================
# MODEL
# ============================================================

model = efficientnet_b0(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support both raw state_dict and checkpoint dictionaries
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# GRAD-CAM
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# EfficientNet-B0 final convolutional feature layer
target_layer = model.features[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ============================================================
# PREDICTION + GRAD-CAM
# ============================================================

def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.zero_grad()

    output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0, predicted_class].item()

    # Backpropagate predicted class
    score = output[0, predicted_class]

    score.backward()

    # Get activations and gradients
    acts = activations.detach()
    grads = gradients.detach()

    # Global average pooling of gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)

    # Weighted feature maps
    cam = (weights * acts).sum(dim=1)

    cam = F.relu(cam)

    # Normalize
    cam = cam.squeeze().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize CAM to original image dimensions
    h, w = original.shape[:2]

    cam = cv2.resize(cam, (w, h))

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert RGB original to BGR
    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    # Overlay
    overlay = cv2.addWeighted(
        original_bgr,
        0.55,
        heatmap,
        0.45,
        0
    )

    # ========================================================
    # LABEL
    # ========================================================

    class_names = ["EDITED", "REAL"]

    label = class_names[predicted_class]

    if label == "EDITED":
        display_label = "AI-EDITED / MANIPULATED"
    else:
        display_label = "REAL"

    # ========================================================
    # SAVE
    # ========================================================

    input_name = Path(image_path).stem

    output_path = OUTPUT_DIR / f"{input_name}_gradcam.jpg"

    cv2.imwrite(
        str(output_path),
        overlay
    )

    print()
    print("=" * 60)
    print("CASIA GRAD-CAM RESULT")
    print("=" * 60)

    print(f"Image: {image_path}")
    print(f"Prediction: {display_label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print()
    print("Class probabilities:")
    print(f"EDITED: {probabilities[0, 0].item() * 100:.2f}%")
    print(f"REAL:   {probabilities[0, 1].item() * 100:.2f}%")

    print()
    print(f"Grad-CAM saved to:")
    print(output_path)

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            'python inference\\gradcam_casia.py "path\\to\\image.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():

        print(f"ERROR: Image not found: {image_path}")

        sys.exit(1)

    generate_gradcam(image_path)