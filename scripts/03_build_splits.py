"""
Step 3: build every split this project uses, from the manifest built in
step 2, and save each one as a JSON file under data/manifest/splits/.

These JSON files get committed to git. Once committed, they never change
for a given result, they only get superseded by a new file if a split
strategy genuinely changes, in which case the old file stays in git
history so any old result can still be checked against the split that
actually produced it.

Usage (from the repository root, after scripts/02_build_manifest.py):
    python scripts/03_build_splits.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.splits import (
    build_country_rotation_folds,
    build_loco_folds,
    build_patient_level_train_val_test,
    build_pooled_baseline_split,
    save_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest/manifest.csv")
    parser.add_argument("--splits_dir", default="data/manifest/splits")
    parser.add_argument("--held_out_group", default="Malawi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    splits_dir = Path(args.splits_dir)

    print("=" * 70)
    print("Building LOCO folds")
    print("=" * 70)
    african = manifest[manifest["source_dataset"] == "african_multicentre"]
    loco = build_loco_folds(african, held_out_group=args.held_out_group, seed=args.seed)
    save_split(loco, splits_dir / "loco_malawi.json")
    for fold in loco["folds"]:
        print(f"  fold {fold['fold_index']}: val country = {fold['val_country']}, "
              f"train patients = {len(fold['train_patient_ids'])}, "
              f"val patients = {len(fold['val_patient_ids'])}")
    print(f"  held out ({args.held_out_group}) patients = "
          f"{len(loco['held_out_patient_ids'])}")

    print()
    print("=" * 70)
    print("Building pooled baseline split")
    print("=" * 70)
    pooled = build_pooled_baseline_split(
        african, held_out_group=args.held_out_group, val_fraction=0.2, seed=args.seed
    )
    save_split(pooled, splits_dir / "pooled_baseline_malawi.json")
    print(f"  train patients = {len(pooled['train_patient_ids'])}, "
          f"val patients = {len(pooled['val_patient_ids'])}")

    print()
    print("=" * 70)
    print("Building country rotation folds")
    print("=" * 70)
    rotation = build_country_rotation_folds(african, seed=args.seed)
    save_split(rotation, splits_dir / "country_rotation.json")
    for entry in rotation["rotation"]:
        print(f"  held out = {entry['held_out_group']}: "
              f"train patients = {len(entry['train_patient_ids'])}, "
              f"test patients = {len(entry['test_patient_ids'])}")

    print()
    print("=" * 70)
    print("Building Spain domain patient level train/val/test split")
    print("=" * 70)
    spain = build_patient_level_train_val_test(
        manifest, group="spain", val_fraction=0.15, test_fraction=0.2, seed=args.seed
    )
    save_split(spain, splits_dir / "spain_patient_level.json")
    print(f"  train patients = {len(spain['train_patient_ids'])}, "
          f"val patients = {len(spain['val_patient_ids'])}, "
          f"test patients = {len(spain['test_patient_ids'])}")

    print()
    print(
        "All splits built and saved. Next step: "
        "python scripts/04_verify_no_leakage.py"
    )


if __name__ == "__main__":
    main()
