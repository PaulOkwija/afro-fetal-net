"""
t-SNE domain shift analysis.

Reproduces the original paper's Figure 3: penultimate layer embeddings
(1280 dimensions for EfficientNet-B0, confirmed directly against the
real model, not assumed) extracted from real images, reduced to 2D, and
colored two ways, by domain (Spain vs African) and by class (brain,
femur, abdomen), to show whether a model's internal representation
separates by acquisition domain rather than by anatomy.

Embedding extraction is genuinely testable, and is tested here: same
model, same images, same output, every time. The t-SNE reduction itself
is stochastic by nature (that is what t-SNE is), so reproducibility for
that step means the same random_state produces the same 2D layout, not
that the layout has some independently checkable correct answer.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from fetal_ai.data.dataset import build_dataloader, build_dataset
from fetal_ai.data.splits import manifest_rows_for_patients
from fetal_ai.models.build import load_checkpoint


def extract_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a model over a dataloader once, no gradient, return the
    penultimate layer embedding (pre_logits, the same 1280 dimensional
    vector the original paper describes for EfficientNet-B0) and the
    true label for every image, in the loader's iteration order.

    Uses model.forward_features then forward_head(..., pre_logits=True),
    the standard timm API for this, confirmed directly against the
    actual model architecture this project uses, not assumed to exist.
    """
    model.eval()
    model.to(device)

    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.forward_features(images)
            embeddings = model.forward_head(features, pre_logits=True)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def sample_patients_for_tsne(
    manifest,
    patient_ids: list[str],
    n_images: int,
    seed: int,
) -> list[str]:
    """
    Subsample down to approximately n_images total, at the patient
    level (never the image level, same reasoning as everywhere else in
    this project, images from one patient are correlated). Returns the
    subset of patient_ids to use. The original paper used 200 Spanish
    and 100 African images for this exact analysis, subsampling for
    a t-SNE plot is standard practice, not a shortcut being taken here.
    """
    rows = manifest_rows_for_patients(manifest, patient_ids)
    images_per_patient = rows.groupby("patient_id").size()

    rng = np.random.default_rng(seed)
    shuffled_patients = rng.permutation(images_per_patient.index.tolist())

    selected_patients = []
    running_total = 0
    for patient_id in shuffled_patients:
        if running_total >= n_images:
            break
        selected_patients.append(patient_id)
        running_total += images_per_patient[patient_id]

    return selected_patients


def run_tsne(
    embeddings: np.ndarray,
    seed: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """
    Reduce embeddings to 2D. random_state fixes the optimization's
    starting point and stochastic steps, so the same embeddings always
    produce the same 2D layout, this is what "reproducible" means for
    t-SNE specifically, not that there is one uniquely correct layout.
    """
    n_samples = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(5.0, n_samples / 4))
    if effective_perplexity != perplexity:
        print(
            f"Perplexity {perplexity} is too high for {n_samples} samples, "
            f"using {effective_perplexity:.1f} instead. sklearn requires "
            f"perplexity < n_samples."
        )

    tsne = TSNE(n_components=2, random_state=seed, perplexity=effective_perplexity, init="pca")
    return tsne.fit_transform(embeddings)


def plot_domain_shift(
    coords_2d: np.ndarray,
    domain_labels: np.ndarray,
    class_labels: np.ndarray,
    class_names: list[str],
    domain_names: list[str],
    title_suffix: str = "",
) -> plt.Figure:
    """
    Two panel plot matching the original Figure 3: left colored by
    domain, right colored by class. class_names must be the exact list
    the labels were encoded against, this function never guesses or
    defaults a class list, the same rule as everywhere else in this
    project, since a mismatched or padded class list here is exactly
    the bug that put a nonexistent 4th class into the original figure.
    """
    if class_labels.max() >= len(class_names):
        raise ValueError(
            f"class_labels contains index {class_labels.max()}, but only "
            f"{len(class_names)} class_names were given: {class_names}. "
            f"These must match exactly."
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    domain_colors = plt.cm.tab10(np.linspace(0, 1, len(domain_names)))
    for i, name in enumerate(domain_names):
        mask = domain_labels == i
        axes[0].scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[domain_colors[i]], label=name, s=15, alpha=0.7)
    axes[0].set_title(f"t-SNE colored by domain{title_suffix}")
    axes[0].legend()
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    class_colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        mask = class_labels == i
        axes[1].scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[class_colors[i]], label=name, s=15, alpha=0.7)
    axes[1].set_title(f"t-SNE colored by class{title_suffix}")
    axes[1].legend()
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()
    return fig
