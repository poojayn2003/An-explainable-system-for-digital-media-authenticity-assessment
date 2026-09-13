import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_best.pth"

OUTPUT_DIR = "outputs/gradcam"

IMG_SIZE = 224

CLASS_NAMES = {
    0: "AI-GENERATED",
    1: "REAL"
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TRUSTIFY AI - GRAD-CAM EXPLAINABILITY")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Trained model not found!")
    print(f"Path: {MODEL_PATH}")
    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    2
)


print("Loading trained model...")

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("✓ Model loaded successfully")


# ============================================================
# TRANSFORM
# ============================================================

imagenet_mean = weights.transforms().mean
imagenet_std = weights.transforms().std

transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        image_tensor,
        target_class
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        activations = self.activations
        gradients = self.gradients

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = torch.relu(cam)

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        cam = cam.squeeze().detach().cpu().numpy()

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam


# ============================================================
# IMAGE PROCESSING
# ============================================================

def load_image(image_path):

    if not os.path.exists(image_path):

        print("\nERROR: Image not found!")
        print(f"Path: {image_path}")

        sys.exit(1)

    image = Image.open(
        image_path
    ).convert("RGB")

    return image


# ============================================================
# GENERATE HEATMAP
# ============================================================

def create_heatmap(
    original_image,
    cam
):

    original = np.array(
        original_image
    )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    height, width = original.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay


# ============================================================
# MAIN PREDICTION + GRAD-CAM
# ============================================================

def analyze_image(image_path):

    print("\n" + "-" * 60)

    print(
        f"Analyzing: {image_path}"
    )

    print("-" * 60)


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image = load_image(
        image_path
    )


    # --------------------------------------------------------
    # Transform image
    # --------------------------------------------------------

    image_tensor = transform(
        original_image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model.zero_grad()

    output = model(
        image_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()


    fake_probability = probabilities[
        0,
        0
    ].item()

    real_probability = probabilities[
        0,
        1
    ].item()


    prediction = CLASS_NAMES[
        predicted_class
    ]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\nRESULT")

    print("=" * 60)

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"FAKE probability : "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"REAL probability : "
        f"{real_probability * 100:.2f}%"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # GRAD-CAM TARGET LAYER
    # --------------------------------------------------------

    target_layer = model.features[-1]


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    print(
        "\nGenerating Grad-CAM..."
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    cam = gradcam.generate(
        image_tensor,
        predicted_class
    )


    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = create_heatmap(
        original_image,
        cam
    )


    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    filename = os.path.basename(
        image_path
    )

    name, _ = os.path.splitext(
        filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_gradcam.jpg"
    )

    cv2.imwrite(
        output_path,
        overlay
    )


    print(
        "\n✓ Grad-CAM generated successfully!"
    )

    print(
        f"Saved to: {output_path}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python inference\\gradcam.py "path\\to\\image.jpg"'
        )

        sys.exit(1)


    image_path = sys.argv[1]

    analyze_image(
        image_path
    )

