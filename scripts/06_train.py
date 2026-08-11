"""
Step 6: train a model.

This is the one script every experiment in this project runs through,
the Spain baseline, every LOCO fold, the pooled baseline, and every
country rotation run. What differs between them is entirely the config
file passed in, never a separate code path. This is the single most
important property for keeping Table 3 and Table 4 in the paper
computed by code that actually agrees with itself, which is the exact
thing that went wrong in the previous version of this project.

Different experiment configs point at different split types (see
src/fetal_ai/data/splits.py), and a split type determines how many
training runs a single config actually produces:

  patient_level_train_val_test (baseline_spain.yaml): one run
  pooled_baseline (pooled_baseline.yaml): one run
  loco (loco_africa.yaml): one run per fold, 4 for this project's data
  country_rotation (country_rotation.yaml): one run per held out country, 5

Every run refuses to proceed on an uncommitted git working tree unless
--allow_dirty is passed, which should only ever be used for local
debugging, never for a run whose numbers will be reported anywhere, see
src/fetal_ai/provenance.py for why.

Usage:
    python scripts/06_train.py --config configs/experiment/baseline_spain.yaml
    python scripts/06_train.py --config configs/experiment/loco_africa.yaml
    python scripts/06_train.py --config configs/experiment/pooled_baseline.yaml
    python scripts/06_train.py --config configs/experiment/country_rotation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.data.dataset import build_dataloader, build_dataset
from fetal_ai.data.splits import load_split
from fetal_ai.models.build import apply_fine_tune_freezing, build_model, load_checkpoint
from fetal_ai.provenance import build_provenance_stamp, save_run_result
from fetal_ai.training.trainer import train_model
from fetal_ai.utils.config import config_hash, load_config
from fetal_ai.utils.seed import set_seed
from fetal_ai.utils.tracking import start_run


def load_data_source_dirs(data_config_path: str) -> tuple[dict[str, str], dict[str, bool]]:
    """
    Read configs/data.yaml to get where each source dataset's images
    live, the same information scripts/02_build_manifest.py and
    scripts/04_verify_no_leakage.py already use. Training never
    hardcodes an image path itself.
    """
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)

    image_dir_by_source = {}
    group_subdir_by_source = {}
    for entry in data_cfg["datasets"].values():
        image_dir_by_source[entry["source_dataset"]] = entry["image_dir"]
        group_subdir_by_source[entry["source_dataset"]] = entry.get("group_subdir", False)

    return image_dir_by_source, group_subdir_by_source


def build_model_for_run(cfg: dict[str, Any], class_names: list[str], device: str):
    """
    Either build a fresh model (Spain baseline, pretrained_checkpoint is
    null) or load a checkpoint and apply fine tuning freezing (every
    African experiment, which starts from the Spain checkpoint).
    """
    model_cfg = cfg["model"]

    if model_cfg.get("pretrained_checkpoint"):
        model, loaded_class_names, _ = load_checkpoint(model_cfg["pretrained_checkpoint"], device=device)
        if loaded_class_names != class_names:
            raise ValueError(
                f"Pretrained checkpoint at {model_cfg['pretrained_checkpoint']} "
                f"has class_names={loaded_class_names}, but this config's "
                f"classes={class_names}. These must match exactly, a "
                f"fine tuning run cannot silently reinterpret what a "
                f"checkpoint's output indices mean."
            )
    else:
        model = build_model(
            architecture=model_cfg["architecture"],
            num_classes=len(class_names),
            pretrained=True,
        )

    apply_fine_tune_freezing(model, fine_tune_layers=model_cfg["fine_tune_layers"])
    return model


def run_one_training(
    cfg: dict[str, Any],
    manifest: pd.DataFrame,
    train_patient_ids: list[str],
    val_patient_ids: list[str],
    class_names: list[str],
    image_dir_by_source: dict[str, str],
    group_subdir_by_source: dict[str, bool],
    device: str,
    run_id: str,
    manifest_path: str,
    allow_dirty: bool,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run exactly one training run: build datasets, build the model,
    train, stamp provenance, save the result. Called once for a single
    split config, or once per fold/rotation entry for LOCO and country
    rotation configs, always through this same function.

    If results/<run_id>/metrics.json already exists, this run is
    skipped and the existing result is loaded and returned instead,
    unless force=True. This exists because a LOCO sweep is 4 training
    runs and a country rotation sweep is 5, and Kaggle sessions have
    time and GPU quota limits, a session dying after fold 2 of 4 should
    not mean folds 0 and 1 get retrained from scratch the next time this
    cell runs. Rerunning the exact same notebook cells after a session
    restart is the intended way to resume a sweep.
    """
    existing_result_path = Path("results") / run_id / "metrics.json"
    if existing_result_path.exists() and not force:
        print(f"\nSkipping {run_id}, already completed, found {existing_result_path}. "
              f"Pass force=True to retrain anyway.")
        with open(existing_result_path) as f:
            return json.load(f)["metrics"]

    print(f"\n{'=' * 70}\nRun: {run_id}\n{'=' * 70}")
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]

    train_dataset = build_dataset(
        manifest=manifest, patient_ids=train_patient_ids, class_names=class_names,
        image_dir_by_source=image_dir_by_source, group_subdir_by_source=group_subdir_by_source,
        image_size=data_cfg["image_size"], is_training=True,
        preprocessing_config=cfg["preprocessing"], augmentation_config=cfg["augmentation"],
    )
    val_dataset = build_dataset(
        manifest=manifest, patient_ids=val_patient_ids, class_names=class_names,
        image_dir_by_source=image_dir_by_source, group_subdir_by_source=group_subdir_by_source,
        image_size=data_cfg["image_size"], is_training=False,
        preprocessing_config=cfg["preprocessing"],
    )

    print(f"Train: {len(train_dataset)} images from {len(train_patient_ids)} patients")
    print(f"Val:   {len(val_dataset)} images from {len(val_patient_ids)} patients")

    train_loader = build_dataloader(train_dataset, batch_size=data_cfg["batch_size"], is_training=True, num_workers=data_cfg["num_workers"])
    val_loader = build_dataloader(val_dataset, batch_size=data_cfg["batch_size"], is_training=False, num_workers=data_cfg["num_workers"])

    model = build_model_for_run(cfg, class_names, device)

    checkpoint_path = Path("results") / run_id / "checkpoint.pt"

    provenance = build_provenance_stamp(
        config_hash=config_hash(cfg), manifest_path=manifest_path,
        seed=cfg["seed"], allow_dirty=allow_dirty,
    )

    tracking_run = start_run(
        project="afro-fetal-net", run_name=run_id, config=cfg, provenance=provenance,
    )

    history = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        class_names=class_names, training_config=cfg["training"], device=device,
        checkpoint_out_path=checkpoint_path, architecture=cfg["model"]["architecture"],
        tracking_run=tracking_run,
    )

    tracking_run.finish()

    save_run_result(run_id=run_id, metrics=history, provenance=provenance)
    print(f"Run {run_id} complete. Best val_f1_macro={history['best_val_f1_macro']:.4f} "
          f"at epoch {history['best_epoch']}. Checkpoint: {checkpoint_path}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment config in configs/experiment/")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--allow_dirty", action="store_true",
        help="Allow running against an uncommitted git working tree. Only "
             "for local debugging, never for a run whose results will be "
             "reported anywhere.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if results/<run_id>/metrics.json already exists. "
             "Default is to skip completed runs, so a sweep can resume "
             "across Kaggle sessions without redoing finished work.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg["data"]["manifest_path"])
    split = load_split(Path(cfg["data"]["splits_dir"]) / f"{cfg['data']['split_name']}.json")
    image_dir_by_source, group_subdir_by_source = load_data_source_dirs(args.data_config)
    class_names = cfg["classes"]

    print(f"Config: {args.config}")
    print(f"Split type: {split['split_type']}")
    print(f"Device: {args.device}")

    common_kwargs = dict(
        cfg=cfg, manifest=manifest, class_names=class_names,
        image_dir_by_source=image_dir_by_source, group_subdir_by_source=group_subdir_by_source,
        device=args.device, manifest_path=cfg["data"]["manifest_path"], allow_dirty=args.allow_dirty,
        force=args.force,
    )

    if split["split_type"] == "patient_level_train_val_test":
        run_one_training(
            train_patient_ids=split["train_patient_ids"], val_patient_ids=split["val_patient_ids"],
            run_id=cfg["experiment_name"], **common_kwargs,
        )

    elif split["split_type"] == "pooled_baseline":
        run_one_training(
            train_patient_ids=split["train_patient_ids"], val_patient_ids=split["val_patient_ids"],
            run_id=cfg["experiment_name"], **common_kwargs,
        )

    elif split["split_type"] == "loco":
        for fold in split["folds"]:
            run_one_training(
                train_patient_ids=fold["train_patient_ids"], val_patient_ids=fold["val_patient_ids"],
                run_id=f"{cfg['experiment_name']}_fold{fold['fold_index']}_{fold['val_country']}",
                **common_kwargs,
            )

    elif split["split_type"] == "country_rotation":
        for entry in split["rotation"]:
            # country rotation has no separate validation set defined,
            # holdout is the test set, so a small slice of train is used
            # for validation and early stopping, done here explicitly
            # rather than silently inside splits.py, since this is a
            # training concern, not a data partitioning one.
            train_ids = entry["train_patient_ids"]
            n_val = max(1, int(len(train_ids) * 0.2))
            val_ids = train_ids[:n_val]
            train_ids_minus_val = train_ids[n_val:]
            run_one_training(
                train_patient_ids=train_ids_minus_val, val_patient_ids=val_ids,
                run_id=f"{cfg['experiment_name']}_heldout_{entry['held_out_group']}",
                **common_kwargs,
            )

    else:
        raise ValueError(f"Unknown split_type: {split['split_type']}")


if __name__ == "__main__":
    main()
