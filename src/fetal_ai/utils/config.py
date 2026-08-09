"""
Strict configuration loading.

The previous version of this project had a module level default,
CLASSES = CLASSES_4C, which quietly meant "4 class" whenever a script
forgot to say otherwise. That default leaked a class that should not
have existed into a figure in the paper. The fix here is not "be more
careful next time." The fix is that this loader refuses to run if a
required field is missing, so there is no default left to leak.

Read this file top to bottom, it is short on purpose.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a config file is missing a field this project requires."""


# Every key listed here must be present in a config before we let a script run.
# Add to this list whenever a new field is introduced that changes what data,
# what classes, or what split gets used. Do not add a field here and also give
# it a default elsewhere, that recreates the exact bug this file exists to stop.
REQUIRED_TOP_LEVEL_KEYS = [
    "experiment_name",
    "classes",
    "data",
    "model",
    "training",
    "seed",
]

REQUIRED_DATA_KEYS = [
    "manifest_path",
    "split_name",
]


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file and validate it.

    Raises ConfigError if any required field is missing. Never fills in a
    default for a required field. If you find yourself wanting to add a
    default here, stop, and instead make the field explicit in every
    config file that needs it.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ConfigError(f"Config file is empty: {path}")

    missing_top = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing_top:
        raise ConfigError(
            f"Config {path} is missing required top level keys: {missing_top}"
        )

    missing_data = [k for k in REQUIRED_DATA_KEYS if k not in cfg["data"]]
    if missing_data:
        raise ConfigError(
            f"Config {path} is missing required data.* keys: {missing_data}"
        )

    if not isinstance(cfg["classes"], list) or len(cfg["classes"]) == 0:
        raise ConfigError(
            f"Config {path} field 'classes' must be a non empty list, "
            f"got: {cfg.get('classes')}"
        )

    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """
    Return a short, stable hash of a config dict.

    This hash gets embedded into every results file, so a reported number
    can always be tied back to the exact config that produced it, not just
    the config file's name (which someone could edit after the fact).
    """
    normalized = yaml.dump(cfg, sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
