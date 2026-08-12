"""
Evaluate a checkpoint against a real test set, with a patient level
bootstrap confidence interval, not a bare point estimate.

This is the one place a trained checkpoint gets scored against held out
data. Training (trainer.py) only ever reports validation metrics during
the training loop, it never touches a true test set. This file is what
actually answers "how good is this model," and it does so the same way
for every checkpoint, the Spain baseline evaluated zero shot on Malawi,
the pooled baseline, the model soup, every country rotation model,
always through this one function, so results are never computed by two
slightly different pieces of code that happen to agree by comment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from fetal_ai.data.dataset import build_dataloader, build_dataset
from fetal_ai.data.splits import manifest_rows_for_patients
from fetal_ai.evaluation.bootstrap import bootstrap_metric_ci
from fetal_ai.evaluation.metrics import compute_classification_metrics
from fetal_ai.models.build import load_checkpoint
from sklearn.metrics import f1_score


def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    patient_ids_in_order: list[str],
    device: str,
) -> pd.DataFrame:
    """
    Run a model over a dataloader once, no gradient, return one row per
    image with patient_id, true label, predicted label, and full
    predicted probability vector. patient_ids_in_order must be in the
    exact same order the dataloader's underlying dataset iterates in,
    since a DataLoader batch carries labels but not patient_id, this is
    how each prediction gets traced back to the patient it belongs to.
    """
    model.eval()
    model.to(device)

    all_y_true, all_y_pred, all_y_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_y_true.append(labels.numpy())
            all_y_pred.append(preds.cpu().numpy())
            all_y_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_probs = np.concatenate(all_y_probs)

    if len(y_true) != len(patient_ids_in_order):
        raise ValueError(
            f"Got {len(y_true)} predictions but {len(patient_ids_in_order)} "
            f"patient IDs, the dataloader must be built with shuffle=False "
            f"(see build_dataloader's is_training=False path) or patient "
            f"IDs cannot be reliably matched back to predictions."
        )

    return pd.DataFrame({
        "patient_id": patient_ids_in_order,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": list(y_probs),
    })


def evaluate_checkpoint(
    checkpoint_path: str,
    manifest: pd.DataFrame,
    patient_ids: list[str],
    image_dir_by_source: dict[str, str],
    group_subdir_by_source: dict[str, bool],
    image_size: int,
    preprocessing_config: dict[str, Any],
    device: str,
    batch_size: int = 32,
    bootstrap_n: int = 2000,
    bootstrap_confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Evaluate one checkpoint against one set of patients, always with a
    patient level bootstrap CI, never a bare point estimate.

    patient_ids should come from a split's held out or test set, for
    example the LOCO split's held_out_patient_ids (the true, unseen
    Malawi test set), never from a train or val split, this function
    does not check that for you, the caller is responsible for passing
    genuinely held out patients.

    Returns point estimates for f1_macro, accuracy, and pr_auc_macro,
    plus a bootstrap CI specifically for f1_macro (the metric this
    project's early stopping and paper reporting are built around,
    see trainer.py). accuracy and pr_auc_macro are reported as point
    estimates only in this first pass, that scope limit is deliberate,
    not an oversight, extending the bootstrap to those metrics is
    straightforward future work if needed.

    Also returns "predictions", one row per image with patient_id,
    filename, true label, and predicted label as strings, not just the
    aggregate metrics. This exists specifically so two evaluation runs
    (for example the same checkpoint with and without CLAHE) can be
    compared image by image, not only by their summary F1, which can
    stay identical even when individual predictions genuinely differ,
    or can differ for reasons that have nothing to do with what changed
    between the two runs. See DECISIONS_LOG.md.
    """
    model, class_names, checkpoint_payload = load_checkpoint(checkpoint_path, device=device)

    rows = manifest_rows_for_patients(manifest, patient_ids)
    if len(rows) == 0:
        raise ValueError(
            f"No manifest rows found for the given {len(patient_ids)} "
            f"patient_ids, cannot evaluate against an empty test set."
        )

    dataset = build_dataset(
        manifest=manifest, patient_ids=patient_ids, class_names=class_names,
        image_dir_by_source=image_dir_by_source, group_subdir_by_source=group_subdir_by_source,
        image_size=image_size, is_training=False, preprocessing_config=preprocessing_config,
    )
    loader = build_dataloader(dataset, batch_size=batch_size, is_training=False, num_workers=0)

    # dataset.rows preserves the exact row order build_dataloader iterates
    # in, since is_training=False means shuffle=False, this order is what
    # ties each prediction back to the patient and file it came from.
    patient_ids_in_order = dataset.rows["patient_id"].tolist()
    filenames_in_order = dataset.rows["filename"].tolist()

    predictions_df = run_inference(model, loader, patient_ids_in_order, device)
    predictions_df["filename"] = filenames_in_order

    y_true = np.array(predictions_df["y_true"].tolist())
    y_pred = np.array(predictions_df["y_pred"].tolist())
    y_probs = np.array(predictions_df["y_probs"].tolist())

    point_estimates = compute_classification_metrics(y_true, y_pred, y_probs, class_names)

    def f1_macro_metric(y_t, y_p):
        return f1_score(y_t, y_p, labels=list(range(len(class_names))), average="macro", zero_division=0)

    f1_ci = bootstrap_metric_ci(
        predictions_df[["patient_id", "y_true", "y_pred"]],
        metric_fn=f1_macro_metric,
        n_bootstrap=bootstrap_n,
        confidence=bootstrap_confidence,
    )

    idx_to_class = {i: name for i, name in enumerate(class_names)}
    predictions = [
        {
            "patient_id": row.patient_id,
            "filename": row.filename,
            "y_true": idx_to_class[int(row.y_true)],
            "y_pred": idx_to_class[int(row.y_pred)],
            "correct": bool(row.y_true == row.y_pred),
        }
        for row in predictions_df.itertuples()
    ]

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_extra": {
            k: v for k, v in checkpoint_payload.items()
            if k not in ("state_dict",)  # never serialize raw weights into a results json
        },
        "class_names": class_names,
        "n_images": len(predictions_df),
        "n_patients": len(set(patient_ids_in_order)),
        "point_estimates": point_estimates,
        "f1_macro_bootstrap_ci": f1_ci,
        "predictions": predictions,
    }
