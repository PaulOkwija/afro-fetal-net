"""
Provenance stamping.

Every result written by this project carries a stamp that answers four
questions without anyone needing to remember anything: what code produced
this, what config, what data, and when. This is how a number in the paper
gets traced back to one exact, reproducible source, instead of a comment
in a notebook that says "this is Malawi" with nothing to check it against.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_git_commit(repo_root: str | Path = ".") -> str:
    """
    Return the current git commit hash.

    Raises if the working tree has uncommitted CODE changes, on purpose.
    A result produced from uncommitted code cannot be traced back to
    anything, so we refuse to stamp it as if it could be.

    data/manifest/ is excluded from this check. That directory holds
    generated data (manifest.csv, splits/*.json), not code, and
    reproducibility of that data does not depend on it being committed
    to git: fetch.py checksum verifies every raw file against Zenodo,
    manifest.py and splits.py are deterministic given the same raw data
    and a fixed seed, and the exact content of whatever manifest a run
    actually used is already captured by data_manifest_hash below,
    computed directly from the file's bytes. Requiring a git commit of
    freshly regenerated data on every Kaggle session added ceremony
    without adding any safety this hash does not already provide, so it
    is not required. Code changes anywhere else still block a run, that
    is the property that actually matters: the code that produced a
    result must be pinned, unambiguously, every time.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)data/manifest"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "Working tree has uncommitted code changes. Commit your code "
            "before running an experiment whose results you plan to "
            "report. This check ignores data/manifest/, which does not "
            "need to be committed, its exact content is already captured "
            "by data_manifest_hash. Uncommitted changed files:\n" + status.stdout
        )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return commit.stdout.strip()


def file_hash(path: str | Path) -> str:
    """Return the sha256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance_stamp(
    config_hash: str,
    manifest_path: str | Path,
    seed: int,
    repo_root: str | Path = ".",
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """
    Build the full provenance dict that gets saved alongside every result.

    allow_dirty exists only for local debugging runs that will never be
    reported. Any run that produces a number for the paper must be run
    with allow_dirty=False, which is also the default in every script.
    """
    if allow_dirty:
        commit = "DIRTY_UNTRACKED_RUN_DO_NOT_CITE"
    else:
        commit = get_git_commit(repo_root)

    return {
        "git_commit": commit,
        "config_hash": config_hash,
        "data_manifest_hash": file_hash(manifest_path),
        "data_manifest_path": str(manifest_path),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def save_run_result(
    run_id: str,
    metrics: dict[str, Any],
    provenance: dict[str, Any],
    results_dir: str | Path = "results",
) -> Path:
    """
    Save metrics plus provenance to results/<run_id>/metrics.json.

    This is the only function anywhere in the codebase that writes a
    metrics.json file. Anything that reports a number for the paper must
    have gone through this function, so the file format never drifts.
    """
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "metrics": metrics,
        "provenance": provenance,
    }

    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path
