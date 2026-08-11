"""
Step 7: model soup, average the LOCO fold checkpoints into one model.

This is the piece that turns four separately fine tuned LOCO checkpoints
into the single model the paper's headline result actually describes.
Nothing before this point produces that model, 06_train.py's LOCO
dispatch only produces the four fold checkpoints individually.

Uniform weight averaging, matching the paper's formula exactly:
theta_soup = (1/M) * sum(theta_m), M = 4 active LOCO fold models.

Refuses to average checkpoints with different class_names or different
architectures, loudly, rather than silently averaging two models that
do not actually agree on what their own output means.

Usage:
    python scripts/07_model_soup.py --loco_config configs/experiment/loco_africa.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.splits import load_split
from fetal_ai.models.build import build_model, load_checkpoint, save_checkpoint
from fetal_ai.provenance import build_provenance_stamp, save_run_result
from fetal_ai.utils.config import config_hash, load_config


def average_state_dicts(state_dicts: list[dict]) -> dict:
    """
    Uniform average of a list of state dicts, all keys must match
    exactly across every state dict, refuses to average mismatched
    architectures rather than silently skipping missing keys.

    Only floating point tensors are actually averaged. Integer buffers,
    the most common example is BatchNorm's num_batches_tracked, a
    training step counter rather than a learnable weight, are copied
    from the first state dict unchanged rather than float-averaged and
    silently truncated back to an integer on load. Averaging a batch
    count across four differently trained folds has no clean meaning,
    and relying on load_state_dict's implicit float-to-int truncation
    to paper over that is not a decision this function should make
    silently. This has no effect on inference correctness either way,
    since num_batches_tracked is not used by BatchNorm in eval mode,
    but it is handled deliberately here rather than by accident.
    """
    keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != keys:
            raise ValueError(
                f"State dict {i} has different keys than state dict 0, "
                f"these checkpoints do not share the same architecture, "
                f"refusing to average them."
            )

    averaged = {}
    non_float_keys_copied = []

    for key in keys:
        reference_tensor = state_dicts[0][key]

        if not torch.is_floating_point(reference_tensor):
            averaged[key] = reference_tensor.clone()
            non_float_keys_copied.append(key)
            continue

        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0).to(reference_tensor.dtype)

    if non_float_keys_copied:
        print(
            f"{len(non_float_keys_copied)} non floating point buffer(s) "
            f"(such as num_batches_tracked) were copied from the first "
            f"checkpoint rather than averaged, this is expected."
        )

    return averaged


def main(argv: list[str] | None = None) -> None:
    """
    argv defaults to None, which makes argparse read sys.argv, correct
    when this script is launched as its own process via `!python
    scripts/07_model_soup.py --loco_config ...`, which is the only
    supported way to run this. If this function is ever called directly
    inside a running notebook kernel instead, sys.argv belongs to the
    kernel launcher, not this script, and parsing it will fail with a
    confusing "unrecognized arguments" error naming a kernel connection
    file. Pass argv explicitly, for example main(["--loco_config",
    "configs/experiment/loco_africa.yaml"]), to bypass sys.argv
    entirely if you genuinely need to call this in-kernel.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loco_config", default="configs/experiment/loco_africa.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_dirty", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.loco_config)
    split = load_split(Path(cfg["data"]["splits_dir"]) / f"{cfg['data']['split_name']}.json")

    if split["split_type"] != "loco":
        raise ValueError(
            f"--loco_config must point at a config whose split_type is "
            f"'loco', got '{split['split_type']}'. Model soup only "
            f"makes sense for LOCO fold checkpoints."
        )

    fold_run_ids = [
        f"{cfg['experiment_name']}_fold{fold['fold_index']}_{fold['val_country']}"
        for fold in split["folds"]
    ]
    print(f"Averaging {len(fold_run_ids)} fold checkpoints:")
    for run_id in fold_run_ids:
        print(f"  {run_id}")

    state_dicts = []
    architectures = set()
    class_names_seen = set()
    fold_val_f1s = []

    for run_id in fold_run_ids:
        checkpoint_path = Path("results") / run_id / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found for {run_id} at {checkpoint_path}. "
                f"Run scripts/06_train.py --config {args.loco_config} "
                f"first, all 4 folds must complete before model soup can run."
            )

        _, class_names, payload = load_checkpoint(checkpoint_path, device="cpu")
        state_dicts.append(payload["state_dict"])
        architectures.add(payload["architecture"])
        class_names_seen.add(tuple(class_names))
        fold_val_f1s.append(payload.get("val_f1_macro"))

    if len(architectures) > 1:
        raise ValueError(f"Fold checkpoints use different architectures: {architectures}")
    if len(class_names_seen) > 1:
        raise ValueError(f"Fold checkpoints use different class_names: {class_names_seen}")

    architecture = architectures.pop()
    class_names = list(class_names_seen.pop())

    print(f"\nAll {len(fold_run_ids)} checkpoints agree: architecture={architecture}, "
          f"classes={class_names}")
    print(f"Fold val_f1_macro values going in: {fold_val_f1s}")

    averaged_state_dict = average_state_dicts(state_dicts)

    model = build_model(architecture, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(averaged_state_dict)

    soup_run_id = f"{cfg['experiment_name']}_model_soup"
    checkpoint_path = Path("results") / soup_run_id / "checkpoint.pt"

    save_checkpoint(
        model, checkpoint_path, class_names=class_names, architecture=architecture,
        extra={"source_fold_run_ids": fold_run_ids, "source_fold_val_f1_macro": fold_val_f1s},
    )

    provenance = build_provenance_stamp(
        config_hash=config_hash(cfg), manifest_path=cfg["data"]["manifest_path"],
        seed=cfg["seed"], allow_dirty=args.allow_dirty,
    )
    save_run_result(
        run_id=soup_run_id,
        metrics={"source_fold_run_ids": fold_run_ids, "source_fold_val_f1_macro": fold_val_f1s,
                 "n_folds_averaged": len(fold_run_ids)},
        provenance=provenance,
    )

    print(f"\nModel soup saved to {checkpoint_path}")
    print(f"Next: evaluate this checkpoint against the Malawi held out test set.")


if __name__ == "__main__":
    main()
