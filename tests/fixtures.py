"""Shared test fixtures. Kept separate from any single test file so that
multiple test modules can use the same synthetic manifest without
importing from each other."""

import pandas as pd


def make_synthetic_manifest() -> pd.DataFrame:
    """
    Build a small fake manifest that mimics the real African dataset's
    shape: 5 countries, 5 patients per country, 3 images per patient,
    3 classes.
    """
    rows = []
    countries = ["Egypt", "Uganda", "Ghana", "Algeria", "Malawi"]
    classes = ["brain", "femur", "abdomen"]

    for country in countries:
        for patient_num in range(5):
            patient_id = f"{country}_{patient_num}"
            for cls in classes:
                rows.append({
                    "patient_id": patient_id,
                    "filename": f"{patient_id}_{cls}.png",
                    "file_sha256": f"fake_hash_{patient_id}_{cls}",
                    "label": cls,
                    "group": country,
                    "source_dataset": "synthetic_test",
                    "original_split": "",
                })

    return pd.DataFrame(rows)
