"""
Step 11: Grad-CAM attention analysis against real data.

Defaults reproduce the original paper's Table 5 and Figure 4: the model
soup checkpoint (the final model), evaluated on the true Malawi held
out test set, one Grad-CAM per image targeted at that image's true
class, summarized per class.

This reports concentration and entropy plainly, as measures of how
sharp or diffuse each heatmap is. It does not claim the maps show
"clinically correct attention," that claim needs expert annotated
landmarks to support, which this project does not have, see
src/fetal_ai/evaluation/gradcam.py's module docstring.

Usage:
    python scripts/11_gradcam_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.dataset import build_dataloader, build_dataset
from fetal_ai.data.splits import load_split
from fetal_ai.evaluation.gradcam import compute_gradcam, make_overlay, summarize_cam_metrics_by_class
from fetal_ai.models.build import load_checkpoint


def load_data_source_dirs(data_config_path: str) -> tuple[dict[str, str], dict[str, bool]]:
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    image_dir_by_source, group_subdir_by_source = {}, {}
    for entry in data_cfg["datasets"].values():
        image_dir_by_source[entry["source_dataset"]] = entry["image_dir"]
        group_subdir_by_source[entry["source_dataset"]] = entry.get("group_subdir", False)
    return image_dir_by_source, group_subdir_by_source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/loco_africa_efficientnet_b0_model_soup/checkpoint.pt")
    parser.add_argument("--manifest", default="data/manifest/manifest.csv")
    parser.add_argument("--held_out_split", default="data/manifest/splits/loco_malawi.json")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--n_examples_per_class", type=int, default=1,
                         help="How many example overlay images to save per class in the figure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_figure", default="results/gradcam_examples.png")
    parser.add_argument("--out_metrics", default="results/gradcam_metrics.json")
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if --out_metrics already exists. Default is to "
             "skip, Grad-CAM over a full test set is not cheap to rerun "
             "repeatedly, and it is deterministic given the same checkpoint.",
    )
    args = parser.parse_args(argv)

    if Path(args.out_metrics).exists() and not args.force:
        print(f"Skipping, {args.out_metrics} already exists. Pass --force to regenerate anyway.")
        return

    manifest = pd.read_csv(args.manifest)
    image_dir_by_source, group_subdir_by_source = load_data_source_dirs(args.data_config)

    split = load_split(args.held_out_split)
    patient_ids = split.get("held_out_patient_ids") or split.get("test_patient_ids")
    if patient_ids is None:
        raise ValueError(f"{args.held_out_split} has neither held_out_patient_ids nor test_patient_ids")

    model, class_names, _ = load_checkpoint(args.checkpoint, device="cpu")
    print(f"Loaded checkpoint: {args.checkpoint}, classes={class_names}")

    dataset = build_dataset(
        manifest=manifest, patient_ids=patient_ids, class_names=class_names,
        image_dir_by_source=image_dir_by_source, group_subdir_by_source=group_subdir_by_source,
        image_size=args.image_size, is_training=False, preprocessing_config={"use_clahe": False},
    )
    loader = build_dataloader(dataset, batch_size=16, is_training=False, num_workers=0)

    print(f"Running Grad-CAM on {len(dataset)} images from {len(patient_ids)} patients")

    all_cams, all_labels, all_images = [], [], []
    for images, labels in loader:
        cams = compute_gradcam(model, images, target_class_indices=labels.tolist())
        all_cams.append(cams)
        all_labels.append(labels.numpy())
        all_images.append(images)

    all_cams = np.concatenate(all_cams)
    all_labels = np.concatenate(all_labels)
    all_images = torch.cat(all_images)

    summary = summarize_cam_metrics_by_class(all_cams, all_labels, class_names)

    print("\nGrad-CAM attention metrics (matches Table 5's structure):")
    print(f"{'Class':<10} {'Conc.':>8} {'C.Std':>8} {'Entropy':>8} {'Entr.Std':>8} {'n':>4}")
    for name, m in summary.items():
        print(f"{name:<10} {m['concentration_mean']:>8.3f} {m['concentration_std']:>8.3f} "
              f"{m['entropy_mean']:>8.3f} {m['entropy_std']:>8.3f} {m['n']:>4}")

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "summary": summary}, f, indent=2)
    print(f"\nMetrics saved to {args.out_metrics}")

    # Save a small figure of real example overlays, n_examples_per_class per class
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(len(class_names), args.n_examples_per_class,
                              figsize=(4 * args.n_examples_per_class, 4 * len(class_names)))
    if args.n_examples_per_class == 1:
        axes = axes.reshape(-1, 1)

    for row, class_name in enumerate(class_names):
        class_indices = np.where(all_labels == row)[0]
        chosen = rng.choice(class_indices, size=min(args.n_examples_per_class, len(class_indices)), replace=False)
        for col, idx in enumerate(chosen):
            overlay = make_overlay(all_images[idx], all_cams[idx])
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(class_name)
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(args.out_figure, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {args.out_figure}")


if __name__ == "__main__":
    main()
