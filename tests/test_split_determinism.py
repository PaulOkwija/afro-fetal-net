"""
A split built with the same seed twice must be identical. A split built
with a different seed should, in general, differ. This matters because
"I re-ran the notebook and got a slightly different number" is exactly
the kind of thing that made the original results hard to trust.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.splits import build_loco_folds, build_pooled_baseline_split
from fixtures import make_synthetic_manifest


def test_loco_folds_are_deterministic_given_same_seed():
    manifest = make_synthetic_manifest()
    result_a = build_loco_folds(manifest, held_out_group="Malawi", seed=42)
    result_b = build_loco_folds(manifest, held_out_group="Malawi", seed=42)
    assert result_a == result_b


def test_pooled_baseline_split_differs_across_seeds():
    manifest = make_synthetic_manifest()
    result_a = build_pooled_baseline_split(
        manifest, held_out_group="Malawi", val_fraction=0.3, seed=1
    )
    result_b = build_pooled_baseline_split(
        manifest, held_out_group="Malawi", val_fraction=0.3, seed=2
    )
    assert result_a["val_patient_ids"] != result_b["val_patient_ids"], (
        "Two different seeds produced the exact same split. Either the "
        "seed is not being used, or this synthetic dataset is too small "
        "for the split to vary, check group count before trusting this."
    )
