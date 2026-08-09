"""
Test that every split-building function in splits.py produces patient
disjoint splits. Uses a small synthetic manifest so this runs in a few
seconds in CI, with no real data and no GPU. This is the test that would
have caught the original Malawi test set problem, if it had existed
before that bug shipped, which is exactly why it exists now.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.splits import (
    assert_patient_disjoint,
    build_country_rotation_folds,
    build_loco_folds,
    build_patient_level_train_val_test,
    build_pooled_baseline_split,
)
from fixtures import make_synthetic_manifest


def test_loco_folds_are_patient_disjoint():
    manifest = make_synthetic_manifest()
    result = build_loco_folds(manifest, held_out_group="Malawi", seed=42)

    assert result["held_out_group"] == "Malawi"
    assert len(result["folds"]) == 4  # 4 non Malawi countries

    for fold in result["folds"]:
        train = set(fold["train_patient_ids"])
        val = set(fold["val_patient_ids"])
        held_out = set(result["held_out_patient_ids"])

        assert train.isdisjoint(val), f"fold {fold['fold_index']} train/val overlap"
        assert train.isdisjoint(held_out), f"fold {fold['fold_index']} train/held_out overlap"
        assert val.isdisjoint(held_out), f"fold {fold['fold_index']} val/held_out overlap"


def test_loco_held_out_group_never_appears_in_any_fold():
    manifest = make_synthetic_manifest()
    result = build_loco_folds(manifest, held_out_group="Malawi", seed=42)

    malawi_patients = set(
        manifest[manifest["group"] == "Malawi"]["patient_id"].unique()
    )

    for fold in result["folds"]:
        all_fold_patients = set(fold["train_patient_ids"]) | set(fold["val_patient_ids"])
        assert malawi_patients.isdisjoint(all_fold_patients), (
            "A Malawi patient appeared inside a LOCO cross validation "
            "fold. This is the exact class of bug this project exists "
            "to prevent."
        )


def test_pooled_baseline_split_is_patient_disjoint():
    manifest = make_synthetic_manifest()
    result = build_pooled_baseline_split(
        manifest, held_out_group="Malawi", val_fraction=0.2, seed=42
    )

    train = set(result["train_patient_ids"])
    val = set(result["val_patient_ids"])
    held_out = set(result["held_out_patient_ids"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(held_out)
    assert val.isdisjoint(held_out)


def test_country_rotation_is_patient_disjoint_for_every_country():
    manifest = make_synthetic_manifest()
    result = build_country_rotation_folds(manifest, seed=42)

    assert len(result["rotation"]) == 5

    for entry in result["rotation"]:
        train = set(entry["train_patient_ids"])
        test = set(entry["test_patient_ids"])
        assert train.isdisjoint(test), (
            f"country rotation for held out group "
            f"'{entry['held_out_group']}' leaked a patient"
        )


def test_patient_level_train_val_test_is_disjoint():
    manifest = make_synthetic_manifest()
    result = build_patient_level_train_val_test(
        manifest, group="Egypt", val_fraction=0.2, test_fraction=0.2, seed=42
    )

    train = set(result["train_patient_ids"])
    val = set(result["val_patient_ids"])
    test = set(result["test_patient_ids"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(
        manifest[manifest["group"] == "Egypt"]["patient_id"].unique()
    )


def test_assert_patient_disjoint_catches_a_real_leak():
    """
    This test intentionally builds a manifest with a leak, and checks
    that assert_patient_disjoint raises. A leakage test that has never
    been shown to actually fail on leaked data is not proven to work.
    """
    manifest = make_synthetic_manifest()

    leaked = manifest.copy()
    leaking_patient = leaked["patient_id"].iloc[0]
    fake_split_col = []
    for _, row in leaked.iterrows():
        if row["patient_id"] == leaking_patient:
            # put half of this patient's rows in "train" and half in "test"
            fake_split_col.append("train" if len(fake_split_col) % 2 == 0 else "test")
        else:
            fake_split_col.append("train")
    leaked["fake_split"] = fake_split_col

    try:
        assert_patient_disjoint(leaked, "fake_split")
        raised = False
    except ValueError:
        raised = True

    assert raised, "assert_patient_disjoint failed to catch a real leak"
