"""
Step 10: t-SNE domain shift analysis, against real data.

Defaults reproduce the original paper's Figure 3 exactly: the Spain
baseline checkpoint (zero shot, no African fine tuning at all),
penultimate layer embeddings from Spain test images and Malawi images,
colored by domain and by class.

Also supports running against the model soup checkpoint, to show the
domain gap after adaptation, a natural companion figure the original
paper did not have, useful evidence for the paper's actual argument.

Usage:
    python scripts/10_tsne_analysis.py

    python scripts/10_tsne_analysis.py \\
        --checkpoint results/loco_africa_efficientnet_b0_model_soup/checkpoint.pt \\
        --out results/tsne_after_adaptation.png --title_suffix " (after adaptation)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.dataset import build_dataloader, build_dataset
from fetal_ai.data.splits import load_split
from fetal_ai.evaluation.tsne import extract_embeddings, plot_domain_shift, run_tsne, sample_patients_for_tsne
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
    parser.add_argument("--checkpoint", default="results/baseline_spain_efficientnet_b0/checkpoint.pt")
    parser.add_argument("--manifest", default="data/manifest/manifest.csv")
    parser.add_argument("--spain_split", default="data/manifest/splits/spain_patient_level.json")
    parser.add_argument("--african_split", default="data/manifest/splits/loco_malawi.json")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--n_spain_images", type=int, default=200, help="Matches the original paper's Figure 3")
    parser.add_argument("--n_african_images", type=int, default=100, help="Matches the original paper's Figure 3")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--title_suffix", default=" (zero shot, before adaptation)")
    parser.add_argument("--out", default="results/tsne_domain_shift.png")
    args = parser.parse_args(argv)

    manifest = pd.read_csv(args.manifest)
    image_dir_by_source, group_subdir_by_source = load_data_source_dirs(args.data_config)

    spain_split = load_split(args.spain_split)
    african_split = load_split(args.african_split)

    spain_patients = sample_patients_for_tsne(
        manifest, spain_split["test_patient_ids"], n_images=args.n_spain_images, seed=args.seed,
    )
    african_patients = sample_patients_for_tsne(
        manifest, african_split["held_out_patient_ids"], n_images=args.n_african_images, seed=args.seed,
    )

    print(f"Sampled {len(spain_patients)} Spain patients, "
          f"{len(african_patients)} African (Malawi) patients")

    model, class_names, _ = load_checkpoint(args.checkpoint, device="cpu")
    print(f"Loaded checkpoint: {args.checkpoint}, classes={class_names}")

    def get_embeddings(patient_ids):
        dataset = build_dataset(
            manifest=manifest, patient_ids=patient_ids, class_names=class_names,
            image_dir_by_source=image_dir_by_source, group_subdir_by_source=group_subdir_by_source,
            image_size=args.image_size, is_training=False, preprocessing_config={"use_clahe": False},
        )
        loader = build_dataloader(dataset, batch_size=32, is_training=False, num_workers=0)
        return extract_embeddings(model, loader, device="cpu")

    spain_embeddings, spain_labels = get_embeddings(spain_patients)
    african_embeddings, african_labels = get_embeddings(african_patients)

    print(f"Spain embeddings: {spain_embeddings.shape}, African embeddings: {african_embeddings.shape}")

    all_embeddings = np.concatenate([spain_embeddings, african_embeddings])
    all_class_labels = np.concatenate([spain_labels, african_labels])
    all_domain_labels = np.concatenate([
        np.zeros(len(spain_embeddings), dtype=int),
        np.ones(len(african_embeddings), dtype=int),
    ])

    print(f"Running t-SNE on {len(all_embeddings)} total embeddings, seed={args.seed}")
    coords_2d = run_tsne(all_embeddings, seed=args.seed, perplexity=args.perplexity)

    fig = plot_domain_shift(
        coords_2d, all_domain_labels, all_class_labels, class_names=class_names,
        domain_names=["spain", "african"], title_suffix=args.title_suffix,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
