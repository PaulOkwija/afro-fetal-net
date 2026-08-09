"""
Build a standardized, checksummed manifest for a raw dataset.

Every dataset this project uses gets turned into one table with the same
column names, regardless of what the original metadata file called
things. This means splits.py never has to know or guess anything about
where a dataset came from, it only ever reads these standard columns:

    patient_id      stable identifier for the patient, used for grouping
    filename        the image filename, relative to the dataset's image dir
    file_sha256     checksum of the actual image bytes on disk
    label           the class label, already mapped to this project's
                     class names (never a raw string like "Fetal brain")
    group           the LOCO grouping variable (country, for the African
                     dataset; a constant "spain" for FETAL_PLANES_DB)
    source_dataset  which of the two source datasets this row came from
    original_split  the train/test split the original authors assigned,
                     kept only for reference, never used to build our own
                     splits (see splits.py for why)

Nothing in this file guesses at a column name. The mapping from a raw
metadata file's actual columns to these standard columns is passed in
explicitly, in a config file, after someone has actually opened the raw
file and looked at it. This is intentional: guessing column names is how
the previous version of this project ended up with a "Malawi" column
that meant different things in two different places.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

STANDARD_COLUMNS = [
    "patient_id",
    "filename",
    "file_sha256",
    "label",
    "group",
    "source_dataset",
    "original_split",
]


def _sha256_of_file(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    raw_metadata_path: str | Path,
    image_dir: str | Path,
    column_mapping: dict[str, str],
    label_mapping: dict[str, str],
    group_value_or_column: str,
    source_dataset: str,
    filename_suffix: str = "",
    csv_separator: str = ",",
    group_subdir: bool = False,
    duplicate_report_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build one standardized manifest dataframe for a single raw dataset.

    column_mapping maps our standard names to the raw file's actual
    column names, for example:
        {"patient_id": "Patient_num", "filename": "Image_name",
         "label": "Plane", "original_split": "Train"}

    label_mapping maps the raw label strings to this project's class
    names, for example {"Fetal brain": "brain", "Fetal thorax": "other"}.
    Keys must match the raw file exactly, including case, this function
    does not normalize case for you, since silently normalizing case is
    how a labeling mismatch goes unnoticed instead of failing loudly.
    Any raw label not present in label_mapping is dropped, with a printed
    count, rather than silently kept or silently discarded without
    anyone knowing. If dropping labels leaves zero rows, that is treated
    as an error, not a warning, see the check right before this function
    returns.

    group_value_or_column is either a literal string (for a dataset that
    has only one group, like "spain" for FETAL_PLANES_DB) or the name of
    a raw column to use as the group (like "Center" for the African
    dataset, which gives country names).

    filename_suffix is appended to each filename before checking it
    against image_dir, for datasets where the metadata file omits the
    file extension.

    csv_separator is passed straight to pandas, for metadata files that
    are not comma separated. Confirm the real separator by opening the
    file, do not guess, a wrong separator produces a single merged
    column name rather than a clean error, which is easy to miss.

    group_subdir, when True, means images are not all in one flat
    image_dir, but organized in a per-group subfolder underneath it,
    for example image_dir/Malawi/<filename>. When True, the row's group
    value (from group_value_or_column) is used as that subfolder name.

    duplicate_report_path, if given, writes a CSV listing every row
    dropped as a byte-identical duplicate of another row, alongside
    which row it duplicated. Investigation into this project's actual
    source data found real duplicate patient entries in the African
    dataset (the same patient's full image set appears under two
    different Patient_num values) and duplicate video frames within
    single patients in FETAL_PLANES_DB. Both are dropped by the
    checksum based dedup below, this report exists so which specific
    rows were affected is auditable and citable in the paper, rather
    than a print statement that scrolls out of the log.
    """
    raw_metadata_path = Path(raw_metadata_path)
    image_dir = Path(image_dir)

    if raw_metadata_path.suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(raw_metadata_path)
    else:
        raw = pd.read_csv(raw_metadata_path, sep=csv_separator)

    raw.columns = [c.strip() for c in raw.columns]

    for std_col, raw_col in column_mapping.items():
        if raw_col not in raw.columns:
            raise ValueError(
                f"Column mapping says '{std_col}' maps to raw column "
                f"'{raw_col}', but that column is not in "
                f"{raw_metadata_path}. Actual columns are: "
                f"{list(raw.columns)}. Open the file and fix the mapping "
                f"in the config, do not guess."
            )

    rows = []
    dropped_labels: dict[str, int] = {}

    for _, r in raw.iterrows():
        raw_label = str(r[column_mapping["label"]]).strip()
        if raw_label not in label_mapping:
            dropped_labels[raw_label] = dropped_labels.get(raw_label, 0) + 1
            continue

        if group_value_or_column in raw.columns:
            group = str(r[group_value_or_column]).strip()
        else:
            group = group_value_or_column

        # patient_id is prefixed with group (country, or "spain") because
        # Patient_num is not guaranteed unique across countries, the
        # African dataset's raw numbering overlaps between countries.
        # Confirmed by direct inspection: without this prefix, 450 raw
        # rows collapsed to 64 unique patient identities instead of the
        # real 127. This prefix is the only thing that makes patient_id
        # safe to use as a grouping key anywhere in this project.
        patient_id = f"{group}_{str(r[column_mapping['patient_id']]).strip()}"

        filename = str(r[column_mapping["filename"]]).strip() + filename_suffix

        if group_subdir:
            image_path = image_dir / group / filename
        else:
            image_path = image_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(
                f"Manifest row references {image_path}, which does not "
                f"exist. The image directory, filename_suffix, or "
                f"group_subdir setting is probably wrong for this "
                f"dataset. Checked group_subdir={group_subdir}."
            )

        rows.append({
            "patient_id": patient_id,
            "filename": filename,
            "file_sha256": _sha256_of_file(image_path),
            "label": label_mapping[raw_label],
            "group": group,
            "source_dataset": source_dataset,
            "original_split": (
                str(r[column_mapping["original_split"]]).strip()
                if "original_split" in column_mapping else ""
            ),
        })

    if dropped_labels:
        print(
            f"Dropped {sum(dropped_labels.values())} rows with labels not "
            f"in label_mapping: {dropped_labels}. This is expected if "
            f"label_mapping intentionally excludes some classes, verify "
            f"that is the case before proceeding."
        )

    if len(rows) == 0:
        raise ValueError(
            f"Every row in {raw_metadata_path} was dropped, none of the "
            f"raw labels matched label_mapping. Raw labels seen: "
            f"{dropped_labels}. label_mapping keys must match the raw "
            f"file exactly, including case. This is an error, not a "
            f"warning, because a manifest with zero rows for an entire "
            f"dataset must never pass silently through to training."
        )

    manifest = pd.DataFrame(rows, columns=STANDARD_COLUMNS)

    dupe_mask = manifest.duplicated(subset=["file_sha256"], keep="first")
    n_dropped = int(dupe_mask.sum())

    if n_dropped > 0:
        # Build the audit report before dropping anything, pairing each
        # dropped row with the row that was kept in its place (same
        # sha256, first occurrence in the file).
        kept_by_sha = (
            manifest[~dupe_mask]
            .drop_duplicates(subset=["file_sha256"])
            .set_index("file_sha256")
        )
        dropped_rows = manifest[dupe_mask].copy()
        dropped_rows["kept_patient_id"] = dropped_rows["file_sha256"].map(
            kept_by_sha["patient_id"]
        )
        dropped_rows["kept_filename"] = dropped_rows["file_sha256"].map(
            kept_by_sha["filename"]
        )

        print(
            f"Dropped {n_dropped} exact duplicate images (same file "
            f"content, detected by checksum). This can mean a real "
            f"duplicate patient entry in the source metadata, or "
            f"repeated video frames within one patient's own scan, see "
            f"the duplicate report for exactly which rows and why."
        )

        if duplicate_report_path is not None:
            out_path = Path(duplicate_report_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dropped_rows[[
                "patient_id", "filename", "label", "group",
                "kept_patient_id", "kept_filename", "file_sha256",
            ]].to_csv(out_path, index=False)
            print(f"Duplicate report written to {out_path}")

    manifest = manifest[~dupe_mask]

    return manifest


def combine_manifests(manifests: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate manifests from multiple source datasets into one table."""
    combined = pd.concat(manifests, ignore_index=True)
    dupe_checksums = combined["file_sha256"].duplicated(keep=False)
    if dupe_checksums.any():
        raise ValueError(
            "Found identical image content (same sha256) appearing in "
            "more than one source dataset. This needs to be investigated "
            "before proceeding, it could mean an accidental overlap "
            "between the two datasets, which would be a leakage risk.\n"
            f"{combined[dupe_checksums][['source_dataset', 'filename', 'file_sha256']]}"
        )
    return combined


def save_manifest(manifest: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    print(f"Saved manifest with {len(manifest)} rows to {out_path}")
    return out_path
