"""
One shared way to log a run to Weights and Biases.

The previous version of this project logged to W&B from inside
notebooks, with the W&B run and the actual result file only connected by
whoever remembered to write down the run name somewhere. This module
makes the connection automatic: every W&B run started through here logs
the exact same provenance stamp (git commit, config hash, data manifest
hash) that gets written into results/<run_id>/metrics.json, so someone
looking at either one can always find the other.

If W&B is not available or not configured (for example, a quick local
smoke test), everything here degrades to printing to stdout instead of
raising, so it never blocks a run that does not care about tracking.
"""

from __future__ import annotations

from typing import Any


def start_run(
    project: str,
    run_name: str,
    config: dict[str, Any],
    provenance: dict[str, Any],
    use_wandb: bool = True,
):
    """
    Start a tracked run. Returns a run object with .log(dict) and
    .finish() methods, either a real wandb run or a small stand in that
    just prints, so calling code never needs an if/else around it.
    """
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print(
                "wandb is not installed, falling back to print only "
                "logging. Install it with pip install wandb if you want "
                "real tracking."
            )
            return _PrintRun(run_name)

        full_config = dict(config)
        full_config["_provenance"] = provenance

        run = wandb.init(project=project, name=run_name, config=full_config)
        return run

    return _PrintRun(run_name)


class _PrintRun:
    """Fallback used when wandb is not installed or not requested. Keeps
    the same .log() / .finish() interface so scripts do not need to
    special case it."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        print(f"[no wandb] starting run: {run_name}")

    def log(self, data: dict[str, Any]) -> None:
        print(f"[no wandb] {self.run_name}: {data}")

    def finish(self) -> None:
        print(f"[no wandb] finished run: {self.run_name}")
