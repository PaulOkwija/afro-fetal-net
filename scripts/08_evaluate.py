"""
Step 8: evaluate a checkpoint against the true, held out Malawi test set.

Defaults are set up to answer the most immediate question, "how does the
model soup do on Malawi," with no arguments needed beyond --checkpoint.
Works against any checkpoint though, the Spain baseline (zero shot), the
pooled baseline, any individual LOCO fold, or the model soup, all
through the exact same evaluation code (src/fetal_ai/evaluation/evaluate.py).

Usage:
    python scripts/08_evaluate.py --checkpoint results/loco_africa_efficientnet_b0_model_soup/checkpoint.pt

    python scripts/08_evaluate.py \\
        --checkpoint results/baseline_spain_efficientnet_b0/checkpoint.pt \\
        --run_id spain_zero_shot_on_malawi

    python scripts/08_evaluate.py \\
        --checkpoint results/baseline_spain_efficientnet_b0/checkpoint.pt \\
        --run_id spain_plus_clahe_on_malawi --use_clahe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.splits import load_split
from fetal_ai.evaluation.evaluate import evaluate_checkpoint
from fetal_ai.provenance import build_provenance_stamp, save_run_result
from fetal_ai.utils.config import config_hash


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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifest/manifest.csv")
    parser.add_argument(
        "--held_out_split", default="data/manifest/splits/loco_malawi.json",
        help="Split file to pull the held out test patient_ids from, "
             "defaults to the LOCO split's held_out_patient_ids, the "
             "true unseen Malawi test set.",
    )
    parser.add_argument(
        "--rotation_held_out_group", default=None,
        help="Required when --held_out_split points at a country_rotation "
             "split (a list of 5 entries, one per country, not a single "
             "flat patient list). Selects which entry's test_patient_ids "
             "to evaluate against, must match that entry's checkpoint, "
             "for example --checkpoint "
             "results/country_rotation_efficientnet_b0_heldout_Algeria/checkpoint.pt "
             "must be paired with --rotation_held_out_group Algeria.",
    )
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--use_clahe", action="store_true",
        help="Apply CLAHE preprocessing at evaluation time, independent "
             "of whether the checkpoint was trained with it. This lets "
             "the same Spain checkpoint be evaluated both zero shot and "
             "zero shot plus CLAHE, matching Table 4's first two rows.",
    )
    parser.add_argument("--contrast_threshold", type=float, default=35)
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--run_id", default=None,
        help="Defaults to '<checkpoint's parent folder name>_eval_on_malawi'",
    )
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument("--bootstrap_confidence", type=float, default=0.95)
    parser.add_argument("--allow_dirty", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    checkpoint_source_run_id = Path(args.checkpoint).parent.name
    run_id = args.run_id or f"{checkpoint_source_run_id}_eval_on_malawi"

    existing = Path("results") / run_id / "metrics.json"
    if existing.exists() and not args.force:
        print(f"Skipping {run_id}, already evaluated, found {existing}. "
              f"Pass --force to re-evaluate anyway.")
        return

    manifest = pd.read_csv(args.manifest)
    split = load_split(args.held_out_split)

    if split.get("split_type") == "country_rotation":
        if args.rotation_held_out_group is None:
            available = [entry["held_out_group"] for entry in split["rotation"]]
            raise ValueError(
                f"{args.held_out_split} is a country_rotation split, a list "
                f"of {len(available)} entries ({available}), not a single "
                f"flat patient list. Pass --rotation_held_out_group naming "
                f"which one, matching the checkpoint being evaluated."
            )
        matching = [e for e in split["rotation"] if e["held_out_group"] == args.rotation_held_out_group]
        if not matching:
            available = [entry["held_out_group"] for entry in split["rotation"]]
            raise ValueError(
                f"--rotation_held_out_group '{args.rotation_held_out_group}' "
                f"not found, available groups in this split: {available}"
            )
        patient_ids = matching[0]["test_patient_ids"]

    elif "held_out_patient_ids" in split:
        patient_ids = split["held_out_patient_ids"]
    elif "test_patient_ids" in split:
        patient_ids = split["test_patient_ids"]
    else:
        raise ValueError(
            f"Split at {args.held_out_split} has neither held_out_patient_ids "
            f"nor test_patient_ids, cannot determine the test set from it."
        )

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Against {len(patient_ids)} held out patients from {args.held_out_split}")
    print(f"CLAHE at eval time: {args.use_clahe}")

    image_dir_by_source, group_subdir_by_source = load_data_source_dirs(args.data_config)

    preprocessing_config = {
        "use_clahe": args.use_clahe,
        "contrast_threshold": args.contrast_threshold,
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_size": (8, 8),
    }

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint, manifest=manifest, patient_ids=patient_ids,
        image_dir_by_source=image_dir_by_source, group_subdir_by_source=group_subdir_by_source,
        image_size=args.image_size, preprocessing_config=preprocessing_config,
        device=args.device, batch_size=args.batch_size,
        bootstrap_n=args.bootstrap_n, bootstrap_confidence=args.bootstrap_confidence,
    )

    print(f"\nn_images={result['n_images']}  n_patients={result['n_patients']}")
    print(f"f1_macro point estimate: {result['point_estimates']['f1_macro']:.4f}")
    ci = result["f1_macro_bootstrap_ci"]
    print(f"f1_macro {int(ci['confidence']*100)}% bootstrap CI: "
          f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]  (n={ci['n_unique_patients']} patients)")
    print(f"accuracy: {result['point_estimates']['accuracy']:.4f}")
    print(f"pr_auc_macro: {result['point_estimates'].get('pr_auc_macro')}")

    provenance = build_provenance_stamp(
        # No experiment config file drives an evaluation run, the CLI
        # arguments themselves are the configuration, so hash exactly
        # those, real provenance rather than a placeholder string.
        config_hash=config_hash(vars(args)), manifest_path=args.manifest,
        seed=0, allow_dirty=args.allow_dirty,
    )
    save_run_result(run_id=run_id, metrics=result, provenance=provenance)
    print(f"\nSaved to results/{run_id}/metrics.json")


if __name__ == "__main__":
    main()
