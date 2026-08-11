"""
Grad-CAM attention analysis.

Reproduces the original paper's Figure 4 and Table 5: Grad-CAM
activations at the conv head layer (model.conv_head, confirmed to
exist on the real model, not assumed), quantified by concentration
(fraction of total activation energy inside the top 20% most active
pixels) and normalized Shannon entropy.

One thing this module deliberately does not do: claim the resulting
maps show "clinically correct attention." Concentration and entropy
measure how sharp or diffuse a heatmap is, not whether it falls on the
correct anatomy, this was one of reviewer bR8N's exact objections to
the original submission, and it is a fair one. Without expert
annotated landmarks to compare against, this module reports the
metrics plainly and stops there. See
src/fetal_ai/explainability's earlier NOTE.md for the same point, made
before this file existed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def compute_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_class_indices: list[int],
) -> np.ndarray:
    """
    Compute Grad-CAM at model.conv_head for a batch of images, one
    target class per image (typically each image's true label, matching
    the original paper's per-class analysis, not necessarily the
    model's prediction).

    Returns an array of shape (N, H, W), values already normalized to
    [0, 1] by the underlying library, confirmed directly against the
    real model before this function was written, not assumed from the
    library's documentation alone.
    """
    if not hasattr(model, "conv_head"):
        raise AttributeError(
            "model has no conv_head layer, this function is written "
            "specifically for the EfficientNet-B0 architecture this "
            "project uses, matching the original paper's stated Grad-CAM "
            "target layer. A different architecture needs a different "
            "target layer chosen deliberately, not guessed."
        )

    model.eval()
    cam = GradCAM(model=model, target_layers=[model.conv_head])
    targets = [ClassifierOutputTarget(idx) for idx in target_class_indices]
    return cam(input_tensor=images, targets=targets)


def compute_concentration(cam: np.ndarray, top_fraction: float = 0.2) -> float:
    """
    Fraction of total CAM activation energy contained in the top
    top_fraction of pixels by value, matching the original paper's
    "energy within top 20% of pixels" definition exactly.

    A single hot pixel with everything else zero gives concentration
    close to 1.0 (nearly all energy in a tiny fraction of pixels,
    capped by top_fraction). A perfectly uniform map gives concentration
    equal to top_fraction (0.2 for the default), since energy is spread
    evenly and the top 20% of pixels can only ever hold 20% of a
    uniform total. Both of these are checked directly in this
    project's tests, not just asserted in this docstring.
    """
    flat = cam.flatten().astype(np.float64)
    total_energy = flat.sum()
    if total_energy <= 0:
        return 0.0

    n_top = max(1, int(len(flat) * top_fraction))
    top_values = np.sort(flat)[-n_top:]
    return float(top_values.sum() / total_energy)


def compute_normalized_entropy(cam: np.ndarray) -> float:
    """
    Shannon entropy of the CAM treated as a probability distribution
    over pixels, normalized by the maximum possible entropy for that
    many pixels (log(N)), so the result is always in [0, 1] regardless
    of image resolution, matching the original paper's "normalized
    Shannon entropy."

    A single hot pixel gives entropy near 0 (fully predictable, not
    diffuse). A perfectly uniform map gives entropy of exactly 1.0
    (maximally diffuse, every pixel equally likely). Both checked
    directly in this project's tests.
    """
    flat = cam.flatten().astype(np.float64)
    total = flat.sum()
    if total <= 0:
        return 0.0

    probs = flat / total
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(len(flat))
    if max_entropy <= 0:
        return 0.0

    return float(entropy / max_entropy)


def summarize_cam_metrics_by_class(
    cams: np.ndarray,
    class_indices: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """
    Per class mean and std of concentration and entropy, matching the
    original paper's Table 5 structure exactly (Conc., C. Std, Entropy,
    Entr. Std, plus a Mean row across all classes).
    """
    concentrations = np.array([compute_concentration(cams[i]) for i in range(len(cams))])
    entropies = np.array([compute_normalized_entropy(cams[i]) for i in range(len(cams))])

    summary = {}
    for class_idx, class_name in enumerate(class_names):
        mask = class_indices == class_idx
        if mask.sum() == 0:
            continue
        summary[class_name] = {
            "concentration_mean": float(concentrations[mask].mean()),
            "concentration_std": float(concentrations[mask].std()),
            "entropy_mean": float(entropies[mask].mean()),
            "entropy_std": float(entropies[mask].std()),
            "n": int(mask.sum()),
        }

    summary["Mean"] = {
        "concentration_mean": float(concentrations.mean()),
        "concentration_std": float(concentrations.std()),
        "entropy_mean": float(entropies.mean()),
        "entropy_std": float(entropies.std()),
        "n": int(len(concentrations)),
    }
    return summary


def unnormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Undo the ImageNet normalization applied in dataset.py, returning a
    float32 HWC array in [0, 1], the format show_cam_on_image expects
    for the background image beneath the heatmap overlay.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1).astype(np.float32)


def make_overlay(image_tensor: torch.Tensor, cam: np.ndarray) -> np.ndarray:
    """One image, its CAM, blended into a single RGB overlay for display."""
    rgb_image = unnormalize_image(image_tensor)
    return show_cam_on_image(rgb_image, cam, use_rgb=True)
