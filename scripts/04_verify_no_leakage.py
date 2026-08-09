"""
Step 4: independently re-verify every split for patient level leakage,
against the real manifest, before spending any GPU time.

This duplicates checks that already run inside splits.py when a split is
built, and duplicates the synthetic tests in tests/test_no_leakage.py.
That duplication is deliberate. The whole reason this project failed to
catch its leakage problem before was that nothing actually checked the
real, final split against the real, final data, end to end, before the
results were reported. This script is that check.

This also re-verifies every file's checksum against the manifest, which
catches the case where the raw data on disk has changed (a re-download,
a different file version on Zenodo, a local edit) since the manifest was
built, without anyone updating the manifest to match.

Usage (from the repository root, after scripts/03_build_splits.py):
    python scripts/04_verify_no_leakage.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.splits import load_split


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(manifest: pd.DataFrame, image_dir_by_source: dict[str, str]) -> list[str]:
    problems = []
    for _, row in manifest.iterrows():
        image_dir = image_dir_by_source.get(row["source_dataset"])
        if image_dir is None:
            problems.append(
                f"No image_dir configured for source_dataset "
                f"'{row['source_dataset']}'"
            )
            continue
        image_path = Path(image_dir) / row["filename"]
        if not image_path.exists():
            problems.append(f"Missing file on disk: {image_path}")
            continue
        actual = _sha256_of_file(image_path)
        if actual != row["file_sha256"]:
            problems.append(
                f"Checksum mismatch for {image_path}: manifest says "
                f"{row['file_sha256']}, file on disk is {actual}"
            )
    return problems


def verify_loco_against_manifest(manifest: pd.DataFrame, split: dict) -> list[str]:
    problems = []
    all_patients = set(manifest["patient_id"].unique())
    held_out = set(split["held_out_patient_ids"])

    unknown = held_out - all_patients
    if unknown:
        problems.append(f"LOCO held out patients not found in manifest: {unknown}")

    for fold in split["folds"]:
        train = set(fold["train_patient_ids"])
        val = set(fold["val_patient_ids"])
        if train & val:
            problems.append(f"Fold {fold['fold_index']}: train/val overlap {train & val}")
        if train & held_out:
            problems.append(f"Fold {fold['fold_index']}: train overlaps held out {train & held_out}")
        if val & held_out:
            problems.append(f"Fold {fold['fold_index']}: val overlaps held out {val & held_out}")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest/manifest.csv")
    parser.add_argument("--splits_dir", default="data/manifest/splits")
    parser.add_argument(
        "--skip_checksums", action="store_true",
        help="Skip re-checksumming every raw image file, faster but weaker check",
    )
    parser.add_argument(
        "--fetal_image_dir", default="data/raw/fetal_planes_db/Images",
    )
    parser.add_argument(
        "--african_image_dir", default="data/raw/african_multicentre/Images",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    splits_dir = Path(args.splits_dir)

    all_problems = []

    if not args.skip_checksums:
        print("Re-checksumming every file in the manifest against disk...")
        image_dir_by_source = {
            "fetal_planes_db": args.fetal_image_dir,
            "african_multicentre": args.african_image_dir,
        }
        checksum_problems = verify_checksums(manifest, image_dir_by_source)
        all_problems.extend(checksum_problems)
        print(f"  {len(checksum_problems)} checksum problem(s) found")
    else:
        print("Skipping checksum re-verification (--skip_checksums was set)")

    print("Verifying LOCO split against manifest...")
    loco = load_split(splits_dir / "loco_malawi.json")
    loco_problems = verify_loco_against_manifest(manifest, loco)
    all_problems.extend(loco_problems)
    print(f"  {len(loco_problems)} problem(s) found")

    print("Verifying pooled baseline split...")
    pooled = load_split(splits_dir / "pooled_baseline_malawi.json")
    train = set(pooled["train_patient_ids"])
    val = set(pooled["val_patient_ids"])
    held_out = set(pooled["held_out_patient_ids"])
    if (train & val) or (train & held_out) or (val & held_out):
        all_problems.append("Pooled baseline split has overlapping patient sets")

    print("Verifying country rotation split...")
    rotation = load_split(splits_dir / "country_rotation.json")
    for entry in rotation["rotation"]:
        train_r = set(entry["train_patient_ids"])
        test_r = set(entry["test_patient_ids"])
        if train_r & test_r:
            all_problems.append(
                f"Country rotation held out '{entry['held_out_group']}' "
                f"has overlapping train/test patients"
            )

    print()
    print("=" * 70)
    if all_problems:
        print(f"FAILED: {len(all_problems)} problem(s) found. Do not train "
              f"or report any result until these are fixed.")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("PASSED: no leakage or checksum problems found. Safe to proceed "
              "to training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
