"""
One-off diagnostic, not part of the regular pipeline. Investigates two
things spotted in the manifest build output:

1. Whether patient_id collides across African countries (Patient_num is
   assigned independently per country, so "1" in Egypt and "1" in Uganda
   are different real people sharing a string).

2. For every image flagged as a byte-identical duplicate, whether that
   is the same physical file referenced by more than one metadata row
   (a data entry issue, benign to drop), or two genuinely different
   files on disk that happen to have identical content (a real data
   quality problem worth understanding before we trust it).

This intentionally does not import from fetal_ai.data.manifest, it
recomputes things from scratch and independently, so it can't just
reproduce the same bug it's trying to check for.
"""

import hashlib
from pathlib import Path

import pandas as pd


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def investigate_african_patient_ids():
    print("=" * 70)
    print("INVESTIGATION 1: does patient_id collide across countries")
    print("=" * 70)

    csv_path = "data/raw/african_multicentre/Zenodo_dataset/African_planes_database.csv"
    df = pd.read_csv(csv_path)

    print(f"Total rows in raw CSV: {len(df)}")
    print()
    print("Patient_num range per country:")
    for country, group in df.groupby("Center"):
        nums = sorted(group["Patient_num"].unique())
        print(f"  {country}: {len(nums)} unique numbers, range {min(nums)} to {max(nums)}")

    naive_unique = df["Patient_num"].astype(str).nunique()
    prefixed_unique = (df["Center"].astype(str) + "_" + df["Patient_num"].astype(str)).nunique()

    print()
    print(f"Unique Patient_num values IGNORING country: {naive_unique}")
    print(f"Unique patient identities WITH country prefix: {prefixed_unique}")
    print(f"(5 countries, up to 25 patients each, so {prefixed_unique} should be close to 125)")
    print()


def investigate_duplicates(dataset_name, csv_path, image_root, filename_col,
                            patient_col, label_col, group_col_or_value,
                            keep_labels, filename_suffix, group_subdir,
                            csv_sep=","):
    print("=" * 70)
    print(f"INVESTIGATION 2: duplicate image content in {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(csv_path, sep=csv_sep)
    df.columns = [c.strip() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        label = str(r[label_col]).strip()
        if label not in keep_labels:
            continue

        filename = str(r[filename_col]).strip() + filename_suffix
        if group_col_or_value in df.columns:
            group = str(r[group_col_or_value]).strip()
        else:
            group = group_col_or_value

        image_path = (
            Path(image_root) / group / filename if group_subdir
            else Path(image_root) / filename
        )
        if not image_path.exists():
            continue

        rows.append({
            "patient_id_raw": str(r[patient_col]).strip(),
            "filename": filename,
            "label": label,
            "group": group,
            "image_path": str(image_path),
        })

    manifest = pd.DataFrame(rows)
    print(f"Rows checked (after label filtering, matching the real pipeline): {len(manifest)}")

    print("Computing checksums...")
    manifest["sha256"] = manifest["image_path"].apply(lambda p: sha256_of_file(Path(p)))

    dupes = manifest[manifest.duplicated(subset=["sha256"], keep=False)].sort_values("sha256")
    print(f"Rows involved in a duplicate sha256 group: {len(dupes)}")
    print()

    if len(dupes) == 0:
        print("No duplicates found.")
        return

    same_path, different_path = 0, 0
    for sha, group in dupes.groupby("sha256"):
        paths = group["image_path"].unique()
        if len(paths) == 1:
            same_path += len(group)
            kind = "SAME file referenced by more than one metadata row"
        else:
            different_path += len(group)
            kind = "DIFFERENT files, byte identical content"

        print(f"sha256={sha[:12]}...  ({kind})")
        print(group[["patient_id_raw", "group", "label", "filename", "image_path"]].to_string(index=False))
        print()

    print("=" * 70)
    print(f"Summary for {dataset_name}:")
    print(f"  same file referenced twice in metadata: {same_path}")
    print(f"  genuinely different files, identical bytes: {different_path}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    investigate_african_patient_ids()

    investigate_duplicates(
        dataset_name="african_multicentre",
        csv_path="data/raw/african_multicentre/Zenodo_dataset/African_planes_database.csv",
        image_root="data/raw/african_multicentre/Zenodo_dataset",
        filename_col="Filename",
        patient_col="Patient_num",
        label_col="Plane",
        group_col_or_value="Center",
        keep_labels={"Fetal abdomen", "Fetal brain", "Fetal femur"},
        filename_suffix=".png",
        group_subdir=True,
        csv_sep=",",
    )

    investigate_duplicates(
        dataset_name="fetal_planes_db",
        csv_path="data/raw/fetal_planes_db/FETAL_PLANES_DB_data.csv",
        image_root="data/raw/fetal_planes_db/Images",
        filename_col="Image_name",
        patient_col="Patient_num",
        label_col="Plane",
        group_col_or_value="spain",
        keep_labels={"Fetal abdomen", "Fetal brain", "Fetal femur"},
        filename_suffix=".png",
        group_subdir=False,
        csv_sep=";",
    )