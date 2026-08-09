"""
Step 0: check that the environment matches what requirements.txt expects.

Run this first, every time, on a new Kaggle session. Kaggle images come
with their own preinstalled package versions, and a version mismatch
that goes unnoticed is exactly the kind of thing that makes a result
hard to reproduce six months later. This script does not try to be
clever about it, it just tells you loudly what does not match, and lets
you decide whether to `pip install -r requirements.txt` before going any
further.

Usage:
    python scripts/00_check_environment.py
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


def read_requirements(path: str = "requirements.txt") -> dict[str, str]:
    pinned = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" not in line:
                continue
            name, version = line.split("==")
            pinned[name.strip()] = version.strip()
    return pinned


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    requirements_path = repo_root / "requirements.txt"

    print(f"Python version: {sys.version}")
    print(f"Checking installed packages against {requirements_path}")
    print()

    pinned = read_requirements(str(requirements_path))
    mismatches = []
    missing = []

    for name, expected_version in pinned.items():
        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue

        if installed_version != expected_version:
            mismatches.append((name, expected_version, installed_version))

    if missing:
        print("MISSING packages (not installed at all):")
        for name in missing:
            print(f"  {name}")
        print()

    if mismatches:
        print("VERSION MISMATCHES (installed differs from requirements.txt):")
        for name, expected, actual in mismatches:
            print(f"  {name}: requirements.txt wants {expected}, found {actual}")
        print()

    if not missing and not mismatches:
        print("Environment matches requirements.txt exactly. Good to proceed.")
    else:
        print(
            "Run: pip install -r requirements.txt --break-system-packages\n"
            "before running any script whose results you plan to report. "
            "A result produced against unpinned or mismatched package "
            "versions cannot be reliably reproduced later."
        )


if __name__ == "__main__":
    main()
