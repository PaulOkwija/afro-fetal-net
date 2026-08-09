"""
Test that the config loader refuses to run on an incomplete config, and
that it never fills in a default for a required field. This is the test
that stands in for the module level CLASSES = CLASSES_4C bug: if someone
tries to reintroduce a silent default, this test should make that
obvious by failing whenever a required field goes missing from a config.
"""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetal_ai.utils.config import ConfigError, config_hash, load_config

VALID_CONFIG = {
    "experiment_name": "test_experiment",
    "classes": ["brain", "femur", "abdomen"],
    "data": {
        "manifest_path": "data/manifest/manifest.csv",
        "split_name": "loco_malawi",
    },
    "model": {"architecture": "efficientnet_b0"},
    "training": {"epochs": 20, "learning_rate": 0.00005},
    "seed": 42,
}


def _write_config(cfg: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def test_valid_config_loads_without_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_config(VALID_CONFIG, Path(tmp))
        cfg = load_config(path)
        assert cfg["experiment_name"] == "test_experiment"


def test_missing_classes_field_raises():
    bad_cfg = {k: v for k, v in VALID_CONFIG.items() if k != "classes"}
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_config(bad_cfg, Path(tmp))
        raised = False
        try:
            load_config(path)
        except ConfigError:
            raised = True
        assert raised, "load_config did not raise on a missing 'classes' field"


def test_empty_classes_list_raises():
    bad_cfg = dict(VALID_CONFIG)
    bad_cfg["classes"] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_config(bad_cfg, Path(tmp))
        raised = False
        try:
            load_config(path)
        except ConfigError:
            raised = True
        assert raised, "load_config did not raise on an empty 'classes' list"


def test_missing_data_subfield_raises():
    bad_cfg = dict(VALID_CONFIG)
    bad_cfg["data"] = {"manifest_path": "data/manifest/manifest.csv"}  # missing split_name
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_config(bad_cfg, Path(tmp))
        raised = False
        try:
            load_config(path)
        except ConfigError:
            raised = True
        assert raised, "load_config did not raise on a missing data.split_name field"


def test_config_hash_is_stable_and_sensitive_to_changes():
    hash_a = config_hash(VALID_CONFIG)
    hash_b = config_hash(VALID_CONFIG)
    assert hash_a == hash_b

    changed_cfg = dict(VALID_CONFIG)
    changed_cfg["seed"] = 43
    hash_c = config_hash(changed_cfg)
    assert hash_a != hash_c, "changing a config value did not change its hash"
