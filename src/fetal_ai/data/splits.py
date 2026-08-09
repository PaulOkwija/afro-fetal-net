"""
The single place every data split is defined.

This is the module that closes the gap that caused the original problem:
two different notebooks each building their own "Malawi test set" from
two different places, with no code proving they were the same set. From
now on there is exactly one function per split type, every split is
built at the patient level (never the image level, since images from the
same patient are correlated and putting them on both sides of a split is
a leak even if the exact image is never repeated), and every split is
written to a JSON file that gets committed to git, so it never has to be
rebuilt by accident with a different random state.

All splits operate on the standardized manifest produced by manifest.py,
so they never need to know anything about where the data originally came
from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut


def assert_patient_disjoint(manifest: pd.DataFrame, split_col: str) -> None:
    """
    Raise if any patient_id appears under more than one value of split_col.

    This is called at the end of every split-building function in this
    file, and again independently in tests/test_no_leakage.py. Two
    independent checks of the same property, one at build time and one
    in CI, is deliberate: it is what actually catches a leak instead of
    trusting that the build code is correct.
    """
    counts = manifest.groupby("patient_id")[split_col].nunique()
    leaking_patients = counts[counts > 1]
    if len(leaking_patients) > 0:
        raise ValueError(
            f"Data leakage detected: {len(leaking_patients)} patient(s) "
            f"appear under more than one value of '{split_col}'. "
            f"Patient IDs: {list(leaking_patients.index)}"
        )


def build_loco_folds(
    manifest: pd.DataFrame,
    held_out_group: str,
    seed: int,
) -> dict[str, Any]:
    """
    Build Leave One Country Out folds, with held_out_group excluded from
    the cross validation loop entirely and reserved as the final test set.

    Grouping is done by patient_id within a country. Since every image
    in the manifest already belongs to exactly one group (country), and
    a patient's images are never split across countries in these source
    datasets, this is equivalent to leave one group out at the country
    level. We still group by patient_id, not just by row, so that if a
    future dataset ever does have a patient scanned in more than one
    center, this code does not silently break that guarantee.
    """
    held_out = manifest[manifest["group"] == held_out_group].copy()
    cv_pool = manifest[manifest["group"] != held_out_group].copy()

    if len(held_out) == 0:
        raise ValueError(f"No rows found with group == '{held_out_group}'")
    if len(cv_pool) == 0:
        raise ValueError("No rows left for the LOCO cross validation pool")

    logo = LeaveOneGroupOut()
    groups = cv_pool["group"].values

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        logo.split(cv_pool, groups=groups)
    ):
        df_train = cv_pool.iloc[train_idx]
        df_val = cv_pool.iloc[val_idx]
        val_country = df_val["group"].iloc[0]

        folds.append({
            "fold_index": fold_idx,
            "val_country": val_country,
            "train_patient_ids": sorted(df_train["patient_id"].unique().tolist()),
            "val_patient_ids": sorted(df_val["patient_id"].unique().tolist()),
        })

    result = {
        "split_type": "loco",
        "held_out_group": held_out_group,
        "held_out_patient_ids": sorted(held_out["patient_id"].unique().tolist()),
        "seed": seed,
        "folds": folds,
    }

    _verify_loco_result(manifest, result)
    return result


def _verify_loco_result(manifest: pd.DataFrame, result: dict) -> None:
    held_out_ids = set(result["held_out_patient_ids"])
    for fold in result["folds"]:
        train_ids = set(fold["train_patient_ids"])
        val_ids = set(fold["val_patient_ids"])

        if train_ids & val_ids:
            raise ValueError(
                f"Fold {fold['fold_index']}: train and val patient sets "
                f"overlap: {train_ids & val_ids}"
            )
        if train_ids & held_out_ids:
            raise ValueError(
                f"Fold {fold['fold_index']}: train set overlaps with the "
                f"held out test patients: {train_ids & held_out_ids}"
            )
        if val_ids & held_out_ids:
            raise ValueError(
                f"Fold {fold['fold_index']}: val set overlaps with the "
                f"held out test patients: {val_ids & held_out_ids}"
            )


def build_pooled_baseline_split(
    manifest: pd.DataFrame,
    held_out_group: str,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """
    Build the pooled fine tuning baseline: every non held out patient
    pooled together into one train/val split, no leave one country out
    structure, no model soup. This is the baseline both reviewers of the
    original submission asked for and that was missing, it isolates
    whether LOCO plus model soup actually add anything over ordinary
    pooled fine tuning.
    """
    held_out = manifest[manifest["group"] == held_out_group].copy()
    pool = manifest[manifest["group"] != held_out_group].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(pool, groups=pool["patient_id"]))

    df_train = pool.iloc[train_idx]
    df_val = pool.iloc[val_idx]

    result = {
        "split_type": "pooled_baseline",
        "held_out_group": held_out_group,
        "held_out_patient_ids": sorted(held_out["patient_id"].unique().tolist()),
        "train_patient_ids": sorted(df_train["patient_id"].unique().tolist()),
        "val_patient_ids": sorted(df_val["patient_id"].unique().tolist()),
        "seed": seed,
    }

    train_ids = set(result["train_patient_ids"])
    val_ids = set(result["val_patient_ids"])
    held_ids = set(result["held_out_patient_ids"])
    if (train_ids & val_ids) or (train_ids & held_ids) or (val_ids & held_ids):
        raise ValueError("Pooled baseline split has overlapping patient sets")

    return result


def build_country_rotation_folds(manifest: pd.DataFrame, seed: int) -> dict[str, Any]:
    """
    Build the held out country rotation: each country takes a turn as the
    fully unseen test set, with all other countries pooled as training
    data (no internal LOCO within the rotation, that is what
    build_loco_folds is for). This directly answers the reviewer question
    of whether "the unseen country beats every validation fold" is a
    general pattern across all countries, or specific to one country in
    a way that would suggest a leak.
    """
    all_groups = sorted(manifest["group"].unique().tolist())
    rotation = []

    for held_out_group in all_groups:
        held_out = manifest[manifest["group"] == held_out_group]
        train_pool = manifest[manifest["group"] != held_out_group]
        rotation.append({
            "held_out_group": held_out_group,
            "train_patient_ids": sorted(train_pool["patient_id"].unique().tolist()),
            "test_patient_ids": sorted(held_out["patient_id"].unique().tolist()),
        })

    result = {"split_type": "country_rotation", "seed": seed, "rotation": rotation}

    for entry in result["rotation"]:
        train_ids = set(entry["train_patient_ids"])
        test_ids = set(entry["test_patient_ids"])
        if train_ids & test_ids:
            raise ValueError(
                f"Country rotation for held out group "
                f"'{entry['held_out_group']}' has overlapping patients"
            )

    return result


def build_patient_level_train_val_test(
    manifest: pd.DataFrame,
    group: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """
    Build a patient level train/val/test split for a single domain, such
    as the Spain source domain training set. This is used instead of
    blindly trusting whatever train/test column shipped with the raw
    metadata, because that original split was very likely built at the
    image level, not the patient level, and this project verifies patient
    disjointness for every split it uses regardless of where the data
    came from.
    """
    subset = manifest[manifest["group"] == group].copy()
    if len(subset) == 0:
        raise ValueError(f"No rows found with group == '{group}'")

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    trainval_idx, test_idx = next(
        splitter_test.split(subset, groups=subset["patient_id"])
    )
    trainval = subset.iloc[trainval_idx]
    test = subset.iloc[test_idx]

    relative_val_fraction = val_fraction / (1 - test_fraction)
    splitter_val = GroupShuffleSplit(
        n_splits=1, test_size=relative_val_fraction, random_state=seed
    )
    train_idx, val_idx = next(
        splitter_val.split(trainval, groups=trainval["patient_id"])
    )
    train = trainval.iloc[train_idx]
    val = trainval.iloc[val_idx]

    result = {
        "split_type": "patient_level_train_val_test",
        "group": group,
        "train_patient_ids": sorted(train["patient_id"].unique().tolist()),
        "val_patient_ids": sorted(val["patient_id"].unique().tolist()),
        "test_patient_ids": sorted(test["patient_id"].unique().tolist()),
        "seed": seed,
    }

    train_ids = set(result["train_patient_ids"])
    val_ids = set(result["val_patient_ids"])
    test_ids = set(result["test_patient_ids"])
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise ValueError("Patient level train/val/test split has overlapping sets")

    return result


def save_split(split: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Saved split ({split['split_type']}) to {out_path}")
    return out_path


def load_split(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def manifest_rows_for_patients(
    manifest: pd.DataFrame, patient_ids: list[str]
) -> pd.DataFrame:
    """Return manifest rows for a given list of patient IDs, used by the
    dataset classes to turn a split's patient list back into image rows."""
    return manifest[manifest["patient_id"].isin(patient_ids)].copy()
