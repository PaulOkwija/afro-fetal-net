"""
Model construction and checkpoint save/load.

One rule drives this file: a checkpoint without class_names is not a
valid checkpoint. The original AfroFetalNet had a module level default,
CLASSES = CLASSES_4C, that silently filled in a 4 class list whenever a
checkpoint didn't carry its own class_names, and that is exactly what
leaked a 4th class into a t-SNE legend in a 3 class experiment. This
file makes that impossible: save_checkpoint requires class_names as an
argument, not a default, and load_checkpoint raises if a checkpoint on
disk doesn't have one, rather than guessing.

Every script that builds or loads a model calls the functions in this
file. Nothing reimplements model construction or checkpoint loading
anywhere else, so there is exactly one place this logic can drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn


class CheckpointError(Exception):
    """Raised when a checkpoint is missing required fields, most often
    class_names. This is a hard stop, not a warning, on purpose."""


def build_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a fresh model from a timm architecture name.

    pretrained=True loads ImageNet weights via timm, used for the Spain
    source domain baseline, which trains from scratch on top of ImageNet.
    For African fine tuning runs, pretrained is irrelevant, the real
    starting point is a Spain checkpoint loaded separately with
    load_checkpoint and apply_fine_tune_freezing below.
    """
    model = timm.create_model(
        architecture, pretrained=pretrained, num_classes=num_classes
    )
    return model


def apply_fine_tune_freezing(model: nn.Module, fine_tune_layers: int) -> nn.Module:
    """
    Freeze all but the last fine_tune_layers parameter groups, plus
    always leave the classification head trainable.

    fine_tune_layers=-1 means nothing is frozen, every parameter trains,
    used for the Spain baseline (configs/experiment/baseline_spain.yaml).

    fine_tune_layers=N (N > 0) freezes everything except the last N
    named parameter tensors and the classifier, used for African LOCO
    and pooled baseline fine tuning, matching the original paper's
    "freeze the first half, update the classification head and the
    final four layers" approach, but made explicit and countable here
    rather than described only in prose.
    """
    if fine_tune_layers == -1:
        for param in model.parameters():
            param.requires_grad = True
        return model

    named_params = list(model.named_parameters())
    n_total = len(named_params)

    for i, (name, param) in enumerate(named_params):
        is_classifier = "classifier" in name or "fc" in name or "head" in name
        is_in_last_n = i >= n_total - fine_tune_layers
        param.requires_grad = is_classifier or is_in_last_n

    return model


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    class_names: list[str],
    architecture: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a model checkpoint. class_names is required, not optional and
    not defaulted, on purpose, see this file's module docstring.
    """
    if not class_names or not isinstance(class_names, list):
        raise CheckpointError(
            f"save_checkpoint called with invalid class_names: "
            f"{class_names!r}. This must be the exact ordered list of "
            f"class names the model's output layer corresponds to, and "
            f"it must be explicit, never a default."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "architecture": architecture,
        "num_classes": len(class_names),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    print(f"Saved checkpoint to {path}, classes={class_names}")
    return path


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    """
    Load a model checkpoint. Returns (model, class_names, full_payload).

    Raises CheckpointError if the checkpoint has no class_names field,
    rather than falling back to any default class list. A checkpoint
    saved by anything other than save_checkpoint in this file, or saved
    before this rule existed, will correctly fail to load here, that is
    the intended behavior, not a bug to work around.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    if "class_names" not in payload:
        raise CheckpointError(
            f"Checkpoint at {path} has no class_names field. This "
            f"project never guesses class names for a checkpoint, see "
            f"REPRODUCIBILITY.md item 2. If this checkpoint predates "
            f"that rule, it needs to be retrained, not loaded with an "
            f"assumed class list."
        )

    class_names = payload["class_names"]
    architecture = payload["architecture"]

    model = build_model(architecture, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)

    return model, class_names, payload
