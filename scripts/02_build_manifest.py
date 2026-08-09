"""
Step 2: build the standardized manifest from the raw metadata files.

This reads each dataset's raw metadata file (the CSV or Excel file that
came down with the images), maps its columns onto this project's
standard manifest columns using the mapping in configs/data.yaml, and
writes one combined, checksummed manifest to data/manifest/manifest.csv.

This manifest, not any folder of images, is what every later script
treats as the ground truth for what data exists and which patient it
belongs to.

Usage (from the repository root, after scripts/01_fetch_data.py):
    python scripts/02_build_manifest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.manifest import build_manifest, combine_manifests, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data_cfg = yaml.safe_load(f)

    manifests = []

    for key, entry in data_cfg["datasets"].items():
        print("=" * 70)
        print(f"Building manifest for: {key}")
        print("=" * 70)

        raw_dir = Path(entry["raw_dir"])
        metadata_path = raw_dir / entry["metadata_file"]
        image_dir = Path(entry["image_dir"])

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. Run "
                f"scripts/01_fetch_data.py first, and confirm "
                f"metadata_file in configs/data.yaml matches the actual "
                f"filename after extraction."
            )

        manifest = build_manifest(
            raw_metadata_path=metadata_path,
            image_dir=image_dir,
            column_mapping=entry["column_mapping"],
            label_mapping=entry["label_mapping"],
            group_value_or_column=entry["group_value_or_column"],
            source_dataset=entry["source_dataset"],
            filename_suffix=entry.get("filename_suffix", ""),
        )

        print(f"\n{key}: {len(manifest)} rows, "
              f"{manifest['patient_id'].nunique()} unique patients")
        print(manifest.groupby(["group", "label"]).size().unstack(fill_value=0))
        print()

        manifests.append(manifest)

    print("=" * 70)
    print("Combining manifests")
    print("=" * 70)
    combined = combine_manifests(manifests)

    print(f"\nCombined manifest: {len(combined)} rows, "
          f"{combined['patient_id'].nunique()} unique patients, "
          f"{combined['source_dataset'].nunique()} source datasets")

    out_path = save_manifest(combined, data_cfg["combined_manifest_out"])
    print(f"\nManifest written to {out_path}")
    print(
        "Next step: python scripts/03_build_splits.py"
    )


if __name__ == "__main__":
    main()
