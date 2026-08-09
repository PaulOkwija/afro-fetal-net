"""
Standard classification metrics, computed the same way everywhere.

Kept deliberately small. Every metric here is a direct call into
scikit-learn, nothing custom, so there is nothing here to get subtly
wrong. The point of centralizing it is not correctness of the math, it
is making sure Table 3 and Table 4 in the paper are never computed by
two slightly different pieces of code that happen to both be called
"f1".
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None,
    class_names: list[str],
) -> dict:
    """
    Compute the standard set of metrics this project reports everywhere.

    y_true, y_pred: integer class indices, shape (n_samples,)
    y_probs: predicted probabilities, shape (n_samples, n_classes), or
             None if not available (some evaluation paths only have
             predicted labels).
    class_names: names in index order, must match len(set(y_true)) or
                 fewer if some classes are absent from this particular
                 evaluation set, sklearn handles that via the labels
                 argument below.
    """
    labels = list(range(len(class_names)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            zero_division=0, output_dict=True,
        ),
    }

    if y_probs is not None:
        try:
            metrics["pr_auc_macro"] = float(
                roc_auc_score(y_true, y_probs, labels=labels, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            # Happens if a class is entirely absent from y_true in this
            # particular evaluation set, which can occur on a small
            # bootstrap resample. Report it explicitly instead of
            # crashing the whole evaluation.
            metrics["pr_auc_macro"] = None
            metrics["pr_auc_error"] = str(e)

    return metrics
