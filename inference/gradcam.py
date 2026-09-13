"""
Trustify AI - Grad-CAM Explainability

Reusable Grad-CAM implementation for:
1. CIFAKE EfficientNet-B0
2. CASIA EfficientNet-B0

The module does NOT load models by itself.
Models are supplied by predict.py.
"""

import os

import cv2
import numpy as np
import torch


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Save forward activations
        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        # Save backward gradients
        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    # --------------------------------------------------------
    # Save activation
    # --------------------------------------------------------

    def save_activation(self, module, input, output):

        self.activations = output

    # --------------------------------------------------------
    # Save gradient
    # --------------------------------------------------------

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    def generate(self, image_tensor, target_class):

        self.model.zero_grad()

        # IMPORTANT:
        # Do not use torch.no_grad() here.
        output = self.model(image_tensor)

        target_score = output[0, target_class]

        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

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

        cam = (
            cam.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        return cam

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    original_image,
    cam,
    alpha=0.45
):
    """
    Create a Grad-CAM heatmap overlay.

    Parameters
    ----------
    original_image : PIL Image or NumPy RGB image
    cam : NumPy array
        Grad-CAM activation map.
    alpha : float
        Heatmap blending strength.

    Returns
    -------
    overlay : NumPy BGR image
    """

    # PIL Image -> NumPy
    if hasattr(original_image, "convert"):
        original = np.array(
            original_image.convert("RGB")
        )
    else:
        original = original_image

    # RGB -> BGR for OpenCV
    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    height, width = original.shape[:2]

    # Resize CAM to original image size
    cam = cv2.resize(
        cam,
        (width, height)
    )

    # Convert to 8-bit
    heatmap = np.uint8(
        255 * cam
    )

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Blend original + heatmap
    overlay = cv2.addWeighted(
        original,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay


# ============================================================
# SAVE HEATMAP
# ============================================================

def save_heatmap(
    original_image,
    cam,
    output_path
):
    """
    Generate and save Grad-CAM overlay.
    """

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    overlay = create_heatmap(
        original_image,
        cam
    )

    success = cv2.imwrite(
        output_path,
        overlay
    )

    if not success:
        raise RuntimeError(
            f"Could not save Grad-CAM image: {output_path}"
        )

    return output_path