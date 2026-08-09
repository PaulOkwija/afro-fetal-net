"""
Fetch raw data directly from its original, cited, DOI backed source.

The previous version of this project downloaded a pre-zipped folder from
Google Drive, with no script anywhere showing how that folder's splits
were built. This module replaces that with a direct download from
Zenodo, using Zenodo's own published checksums to verify every file
before it is used. Nothing about the raw data comes from a source that
cannot be independently checked by anyone who reads this file.

Both datasets used in this project are on Zenodo with permanent DOIs:

  FETAL_PLANES_DB (Burgos-Artizzu et al. 2020)
    DOI: 10.5281/zenodo.3904280
    record id: 3904280

  African multi-centre fetal ultrasound dataset (Sendra-Balcells et al. 2023)
    DOI: 10.5281/zenodo.7540448
    record id: 7540448
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

ZENODO_API = "https://zenodo.org/api/records"


def get_zenodo_record_metadata(record_id: str) -> dict[str, Any]:
    """Fetch the file list and checksums for a Zenodo record."""
    url = f"{ZENODO_API}/{record_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _md5_of_file(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zenodo_record(
    record_id: str,
    dest_dir: str | Path,
    verify_checksum: bool = True,
) -> list[Path]:
    """
    Download every file in a Zenodo record to dest_dir, verifying each
    file's md5 checksum against the one Zenodo itself reports.

    Returns the list of downloaded file paths. Raises if a checksum does
    not match, rather than continuing with a file that might be corrupt
    or a different version than expected.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_zenodo_record_metadata(record_id)
    files = metadata.get("files", [])
    if not files:
        raise RuntimeError(
            f"Zenodo record {record_id} returned no files. Check the "
            f"record id and that the record is public."
        )

    downloaded = []
    for file_entry in files:
        filename = file_entry["key"]
        download_url = file_entry["links"]["self"]
        expected_checksum = file_entry.get("checksum", "")
        if expected_checksum.startswith("md5:"):
            expected_checksum = expected_checksum[len("md5:"):]

        out_path = dest_dir / filename

        if out_path.exists() and verify_checksum:
            if _md5_of_file(out_path) == expected_checksum:
                print(f"Already downloaded and verified: {filename}")
                downloaded.append(out_path)
                continue

        print(f"Downloading: {filename}")
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        if verify_checksum:
            actual = _md5_of_file(out_path)
            if actual != expected_checksum:
                raise RuntimeError(
                    f"Checksum mismatch for {filename}. "
                    f"Expected {expected_checksum}, got {actual}. "
                    f"The download may be corrupt or the file changed on "
                    f"Zenodo. Delete {out_path} and try again."
                )
            print(f"Checksum verified: {filename}")

        downloaded.append(out_path)

    return downloaded
