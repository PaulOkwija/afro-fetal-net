"""
Turns a split's patient list into an actual PyTorch Dataset that
trainer.py can consume.

This is the one place image loading, selective CLAHE, and augmentation
happen. Nothing else in this project reads an image file from disk
directly, everything goes through here, so there is exactly one place
this logic can drift between experiments.

One honest note on an underspecified detail: the original paper
describes CLAHE being applied "selectively to images with a global
contrast score below 35 (the lower quartile of the African dataset)"
but never defines exactly what "contrast score" means numerically. This
file defines it as the standard deviation of pixel intensity (RMS
contrast), a standard, common definition, and says so here rather than
presenting it as if it were confirmed against the original
implementation. If the original authors' exact definition ever surfaces,
this is the only place that needs to change.

CLAHE is a training time augmentation here, applied randomly each
epoch, gated by augmentation_config's clahe_p, exactly like the other
augmentations (flips, rotation, jitter, gaussian noise). It never
applies during evaluation, is_training controls that the same way it
gates every other augmentation. This matches how CLAHE was actually
used in the original experiments, confirmed directly rather than
assumed, see DECISIONS_LOG.md. An earlier version of this file applied
CLAHE deterministically at both train and eval time, driven by a
preprocessing_config dict instead. That version is what let the African
fine tuning configs silently contradict the paper's own stated
methodology ("Selective CLAHE is omitted during fine-tuning
augmentations"), since deterministic-always-on CLAHE was quietly
running during every African training run regardless of that stated
intent.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from fetal_ai.data.splits import manifest_rows_for_patients, resolve_image_path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AddGaussianNoise:
    """Adds gaussian noise to a tensor image, applied with probability p.
    torchvision has no built in transform for this, so it is written
    here, used only by the training augmentation pipeline."""

    def __init__(self, p: float, std: float = 0.05):
        self.p = p
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return tensor + noise
        return tensor


def compute_contrast_score(image_gray: np.ndarray) -> float:
    """
    RMS contrast: the standard deviation of pixel intensity values,
    computed on a 0 to 255 grayscale image. See this file's module
    docstring for why this specific definition was chosen.
    """
    return float(np.std(image_gray))


def apply_selective_clahe(
    image_gray: np.ndarray,
    contrast_threshold: float,
    clip_limit: float,
    tile_size: tuple[int, int],
) -> np.ndarray:
    """
    Apply CLAHE only if the image's contrast score is below
    contrast_threshold, otherwise return the image unchanged. Matches
    the paper's description of selective application "to boost
    structural detail without introducing noise artifacts" on images
    that need it, rather than every image.
    """
    score = compute_contrast_score(image_gray)
    if score >= contrast_threshold:
        return image_gray

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_size))
    return clahe.apply(image_gray)


class FetalPlaneDataset(Dataset):
    """
    One row per image. Reads the file, optionally applies selective
    CLAHE, optionally applies training augmentation, always resizes and
    normalizes with ImageNet statistics (every backbone in this project
    starts from ImageNet pretrained weights, see configs/experiment/*.yaml).

    class_names fixes the label index mapping, in the exact order the
    model's output layer uses. This must be the same list passed to
    save_checkpoint, so predictions and checkpoints always agree on what
    index means what class, see src/fetal_ai/models/build.py.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        class_names: list[str],
        image_dir_by_source: dict[str, str],
        group_subdir_by_source: dict[str, bool],
        image_size: int,
        is_training: bool,
        preprocessing_config: dict[str, Any],
        augmentation_config: dict[str, Any] | None = None,
    ):
        self.rows = rows.reset_index(drop=True)
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.image_dir_by_source = image_dir_by_source
        self.group_subdir_by_source = group_subdir_by_source
        self.image_size = image_size
        self.is_training = is_training
        self.preprocessing_config = preprocessing_config
        self.augmentation_config = augmentation_config or {}

        unknown_labels = set(self.rows["label"]) - set(class_names)
        if unknown_labels:
            raise ValueError(
                f"Dataset rows contain labels not in class_names: "
                f"{unknown_labels}. class_names must cover every label "
                f"actually present in the rows passed in, this is not "
                f"silently filtered."
            )

        self._build_transform_pipeline()

    def _build_transform_pipeline(self) -> None:
        ops = [transforms.Resize((self.image_size, self.image_size))]

        if self.is_training and self.augmentation_config:
            cfg = self.augmentation_config
            if "random_horizontal_flip_p" in cfg:
                ops.append(transforms.RandomHorizontalFlip(p=cfg["random_horizontal_flip_p"]))
            if "random_rotation_degrees" in cfg:
                ops.append(transforms.RandomApply(
                    [transforms.RandomRotation(degrees=cfg["random_rotation_degrees"])],
                    p=cfg.get("brightness_contrast_p", 0.7),
                ))
            if "brightness_contrast_p" in cfg:
                ops.append(transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2)],
                    p=cfg["brightness_contrast_p"],
                ))

        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

        if self.is_training and "gaussian_noise_p" in self.augmentation_config:
            ops.append(AddGaussianNoise(p=self.augmentation_config["gaussian_noise_p"]))

        self.transform = transforms.Compose(ops)

        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[idx]
        image_path = resolve_image_path(row, self.image_dir_by_source, self.group_subdir_by_source)

        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(
                f"Could not read image at {image_path}, row was "
                f"patient_id={row['patient_id']}, filename={row['filename']}"
            )

        # CLAHE is a training time augmentation, not an inference time
        # preprocessing step, matching how it was actually used in the
        # original experiments: applied randomly each epoch, alongside
        # the other augmentations, only during training. It is never
        # applied when is_training is False, evaluation always sees the
        # raw image. See DECISIONS_LOG.md for why this changed from an
        # earlier version where CLAHE applied deterministically at both
        # train and eval time, which does not match how it was actually
        # used and, worse, had been silently contradicting the paper's
        # own stated methodology for the African fine tuning configs.
        if self.is_training and "clahe_p" in self.augmentation_config:
            if torch.rand(1).item() < self.augmentation_config["clahe_p"]:
                image_gray = apply_selective_clahe(
                    image_gray,
                    contrast_threshold=self.augmentation_config.get("contrast_threshold", 35),
                    clip_limit=self.augmentation_config.get("clahe_clip_limit", 2.0),
                    tile_size=self.augmentation_config.get("clahe_tile_size", (8, 8)),
                )

        # Ultrasound images are single channel, ImageNet backbones expect
        # 3 channels, replicate the grayscale channel rather than
        # discarding two thirds of the pretrained weights' input filters.
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        image_pil = Image.fromarray(image_rgb)

        image_tensor = self.transform(image_pil)
        label_idx = self.class_to_idx[row["label"]]

        return image_tensor, label_idx


def build_dataset(
    manifest: pd.DataFrame,
    patient_ids: list[str],
    class_names: list[str],
    image_dir_by_source: dict[str, str],
    group_subdir_by_source: dict[str, bool],
    image_size: int,
    is_training: bool,
    preprocessing_config: dict[str, Any],
    augmentation_config: dict[str, Any] | None = None,
) -> FetalPlaneDataset:
    """Build a dataset from a split's patient_ids list, the standard way
    every script in this project turns a split into training data."""
    rows = manifest_rows_for_patients(manifest, patient_ids)
    if len(rows) == 0:
        raise ValueError(
            f"No manifest rows found for the given patient_ids "
            f"(n={len(patient_ids)}). Either the patient_ids list is "
            f"empty, or it does not match this manifest, do not proceed "
            f"with an empty dataset."
        )
    return FetalPlaneDataset(
        rows=rows, class_names=class_names,
        image_dir_by_source=image_dir_by_source,
        group_subdir_by_source=group_subdir_by_source,
        image_size=image_size, is_training=is_training,
        preprocessing_config=preprocessing_config,
        augmentation_config=augmentation_config,
    )


def build_dataloader(
    dataset: FetalPlaneDataset,
    batch_size: int,
    is_training: bool,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    seed controls two things that PyTorch does not make reproducible by
    default once num_workers > 0: the shuffle order (via an explicit
    generator, rather than relying on whatever the ambient global random
    state happens to be at iteration time) and each worker process's own
    random state (via worker_init_fn), which controls the actual
    augmentation randomness (flips, rotation, jitter, gaussian noise)
    applied inside FetalPlaneDataset.__getitem__.

    Without this, num_workers > 0 (every config in this project uses
    num_workers: 4) can produce a different augmented image, and
    therefore different training dynamics, on every run, even with
    set_seed() called and the same config, since forked worker processes
    do not automatically inherit a seed reproducibly from the main
    process. This was never exercised by this project's own tests
    before, which all used num_workers=0, see DECISIONS_LOG.md.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        # torch.initial_seed() inside a worker is derived deterministically
        # from the DataLoader's own base_seed (set from `generator` above)
        # combined with the worker id, so this differs per worker but is
        # exactly reproducible run to run given the same seed.
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_training,
        num_workers=num_workers, drop_last=False,
        generator=generator, worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
