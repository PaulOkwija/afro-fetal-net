"""
The single training loop this project uses, for every experiment.

The original AfroFetalNet problem this whole rebuild responds to was
two different notebooks each training or evaluating a model in slightly
different code, only agreeing by comment that they were doing the same
thing. This file is how that becomes structurally impossible here: the
Spain baseline, every LOCO fold, the pooled baseline, and every country
rotation run all call train_model, with a config controlling what
differs between them. If a bug exists in the training loop, it affects
every experiment identically, which is a testable, traceable property.
No experiment gets its own copy of this logic.

Early stopping is done on validation macro F1, not validation loss,
matching the original paper's stated methodology ("early stopping on
validation macro F1"), so results stay comparable to what was reported
before.

This module does not know how data gets loaded. It takes already built
PyTorch DataLoaders. That is deliberate: this file has one job, run the
training loop correctly, and stays decoupled from CLAHE, augmentation,
and manifest logic, which live in src/fetal_ai/data/.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fetal_ai.evaluation.metrics import compute_classification_metrics
from fetal_ai.models.build import save_checkpoint


def _run_one_epoch_train(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _run_one_epoch_eval(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_y_true = []
    all_y_pred = []
    all_y_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_y_true.append(labels.cpu().numpy())
            all_y_pred.append(preds.cpu().numpy())
            all_y_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_probs = np.concatenate(all_y_probs)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, y_true, y_pred, y_probs


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    training_config: dict[str, Any],
    device: str,
    checkpoint_out_path: str | Path,
    architecture: str,
    tracking_run: Any = None,
) -> dict[str, Any]:
    """
    Train model, early stopping on validation macro F1.

    training_config must contain: epochs, learning_rate, weight_decay,
    early_stopping_patience. optimizer and scheduler are always AdamW
    and cosine annealing respectively, matching the original paper and
    every experiment config in this project, so this is not read from
    training_config, if that ever needs to vary, it should become an
    explicit config field rather than an assumed default.

    tracking_run, if given, must have a .log(dict) method, see
    fetal_ai.utils.tracking.start_run. Passed in rather than created
    here so this function stays testable without needing W&B.

    Returns a history dict: per epoch train/val loss and val f1_macro,
    plus which epoch was best and that epoch's full metrics. The best
    checkpoint (by validation macro F1) is what gets saved to
    checkpoint_out_path, not necessarily the final epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=training_config["epochs"])

    num_classes = len(class_names)
    patience = training_config["early_stopping_patience"]

    history = {"epochs": []}
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improvement = 0

    for epoch in range(training_config["epochs"]):
        start_time = time.time()

        train_loss = _run_one_epoch_train(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred, y_probs = _run_one_epoch_eval(
            model, val_loader, criterion, device, num_classes
        )
        scheduler.step()

        val_metrics = compute_classification_metrics(y_true, y_pred, y_probs, class_names)
        val_f1 = val_metrics["f1_macro"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": val_f1,
            "seconds": time.time() - start_time,
        }
        history["epochs"].append(epoch_record)

        if tracking_run is not None:
            tracking_run.log({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_f1_macro": val_f1,
            })

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1_macro={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_since_improvement = 0
            save_checkpoint(
                model, checkpoint_out_path, class_names=class_names,
                architecture=architecture,
                extra={"epoch": epoch, "val_f1_macro": val_f1, "val_metrics": val_metrics},
            )
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}, no improvement in "
                f"val_f1_macro for {patience} epochs. Best was epoch "
                f"{best_epoch} with val_f1_macro={best_val_f1:.4f}."
            )
            break

    history["best_epoch"] = best_epoch
    history["best_val_f1_macro"] = best_val_f1
    history["checkpoint_path"] = str(checkpoint_out_path)

    return history
