"""
Step 1: fetch both raw datasets directly from Zenodo.

This replaces the previous version's gdown download of an opaque,
pre-zipped Google Drive folder. Every file downloaded here comes
straight from the dataset's own permanent DOI, and every file's
checksum is verified against the one Zenodo itself reports, before
anything downstream is allowed to use it.

Usage (from the repository root):
    python scripts/01_fetch_data.py
    python scripts/01_fetch_data.py --dataset fetal_planes_db
    python scripts/01_fetch_data.py --dataset african_multicentre
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.fetch import download_zenodo_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/data.yaml",
        help="Path to the data source config file",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Fetch only this dataset key from the config, default is all",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        data_cfg = yaml.safe_load(f)

    dataset_keys = (
        [args.dataset] if args.dataset else list(data_cfg["datasets"].keys())
    )

    for key in dataset_keys:
        entry = data_cfg["datasets"][key]
        print("=" * 70)
        print(f"Fetching dataset: {key}")
        print(f"  Zenodo record id: {entry['zenodo_record_id']}")
        print(f"  DOI: {entry['doi']}")
        print(f"  Destination: {entry['raw_dir']}")
        print("=" * 70)

        downloaded_files = download_zenodo_record(
            record_id=entry["zenodo_record_id"],
            dest_dir=entry["raw_dir"],
            verify_checksum=True,
        )

        print(f"\nDownloaded and verified {len(downloaded_files)} file(s) for {key}.")
        print(
            "If any of these are zip archives, extract them now, then "
            "check configs/data.yaml's metadata_file and image_dir "
            "fields for this dataset actually match what came out of "
            "the archive. Do not guess, open the extracted folder and "
            "look."
        )
        print()


if __name__ == "__main__":
    main()
