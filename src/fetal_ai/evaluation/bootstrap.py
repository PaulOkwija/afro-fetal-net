"""
Bootstrap confidence intervals, computed at the patient level.

The original AfroFetalNet paper reported 98.7 percent F1 on 75 images
with no confidence interval at all. On a test set this small, that
single number tells a reader almost nothing about how much it would
move on a different sample of patients. This module fixes that.

The bootstrap resampling here is done over patients, not over images.
If we resampled over images, a patient with 4 images would effectively
get resampled 4 times per patient, understating the true uncertainty,
since those 4 images are correlated (same patient, same scanner,
same day). Resampling over patients and taking all of that patient's
images together in each bootstrap draw is the correct unit of
resampling for this data.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def bootstrap_metric_ci(
    predictions_df: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute a bootstrap confidence interval for a metric, resampling
    patients with replacement.

    predictions_df must have columns: patient_id, y_true, y_pred
    (one row per image, patient_id repeated for patients with multiple
    images).

    metric_fn takes (y_true, y_pred) arrays and returns a single float,
    for example sklearn.metrics.f1_score with average="macro".

    Returns the point estimate (computed on the full data, not the
    bootstrap mean, which is standard practice), the lower and upper
    bounds of the interval, and the number of unique patients that went
    into the estimate, since that number is what actually determines how
    wide the interval will be.
    """
    required_cols = {"patient_id", "y_true", "y_pred"}
    missing = required_cols - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df is missing columns: {missing}")

    point_estimate = metric_fn(
        predictions_df["y_true"].values, predictions_df["y_pred"].values
    )

    unique_patients = predictions_df["patient_id"].unique()
    n_patients = len(unique_patients)

    if n_patients < 10:
        print(
            f"Warning: only {n_patients} unique patients. Bootstrap "
            f"confidence intervals become unreliable below roughly 10 "
            f"independent units, see Davison and Hinkley on small sample "
            f"bootstrap behavior. Report this interval with that caveat, "
            f"and treat it as a lower bound on the true uncertainty, not "
            f"a precise estimate of it."
        )

    rng = np.random.default_rng(seed)
    bootstrap_scores = []

    grouped = {
        pid: df for pid, df in predictions_df.groupby("patient_id")
    }

    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_rows = pd.concat([grouped[pid] for pid in sampled_patients])
        try:
            score = metric_fn(
                sampled_rows["y_true"].values, sampled_rows["y_pred"].values
            )
        except Exception:
            # A bootstrap draw can occasionally miss a class entirely
            # when patient counts per class are small. Skip that draw
            # rather than let a single missing class break the whole
            # interval, but note how often this happened.
            continue
        bootstrap_scores.append(score)

    if len(bootstrap_scores) < n_bootstrap * 0.5:
        print(
            f"Warning: only {len(bootstrap_scores)} of {n_bootstrap} "
            f"bootstrap draws were usable, the rest hit a missing class. "
            f"This is a sign the test set is too small and too imbalanced "
            f"for this metric to be estimated reliably by patient level "
            f"bootstrap. Report this explicitly rather than the interval "
            f"alone."
        )

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "n_bootstrap_used": len(bootstrap_scores),
        "n_bootstrap_requested": n_bootstrap,
        "n_unique_patients": int(n_patients),
        "seed": seed,
    }
