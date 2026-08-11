"""
Step 9: collect whatever results actually exist into one place.

Reads every results/*/metrics.json on disk right now and assembles them
into one table. This never invents a row for an experiment that hasn't
run yet, if a checkpoint or evaluation doesn't exist, it simply doesn't
appear, rather than showing a placeholder that could be mistaken for a
real number.

Two kinds of run_id show up here: training runs (have best_val_f1_macro,
from trainer.py) and evaluation runs (have point_estimates and
f1_macro_bootstrap_ci, from evaluate.py, scripts/08_evaluate.py). Both
get listed, clearly labeled by kind, since a validation score and a true
held out test score answer different questions and should never be
confused for each other in the same column.

Usage:
    python scripts/09_collect_results.py
    python scripts/09_collect_results.py --results_dir results --out results/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def classify_and_extract(run_id: str, metrics: dict[str, Any]) -> dict[str, Any] | None:
    """
    Figure out whether a metrics.json is a training run or an evaluation
    run, and pull out the fields worth summarizing. Returns None for
    anything that matches neither shape, rather than guessing.
    """
    if "best_val_f1_macro" in metrics:
        return {
            "run_id": run_id,
            "kind": "training",
            "val_f1_macro": metrics.get("best_val_f1_macro"),
            "best_epoch": metrics.get("best_epoch"),
            "f1_macro_point": None,
            "f1_macro_ci_lower": None,
            "f1_macro_ci_upper": None,
            "n_test_patients": None,
        }

    if "f1_macro_bootstrap_ci" in metrics:
        ci = metrics["f1_macro_bootstrap_ci"]
        return {
            "run_id": run_id,
            "kind": "evaluation",
            "val_f1_macro": None,
            "best_epoch": None,
            "f1_macro_point": metrics["point_estimates"].get("f1_macro"),
            "f1_macro_ci_lower": ci.get("ci_lower"),
            "f1_macro_ci_upper": ci.get("ci_upper"),
            "n_test_patients": metrics.get("n_patients"),
        }

    if "n_folds_averaged" in metrics:
        return {
            "run_id": run_id,
            "kind": "model_soup",
            "val_f1_macro": None,
            "best_epoch": None,
            "f1_macro_point": None,
            "f1_macro_ci_lower": None,
            "f1_macro_ci_upper": None,
            "n_test_patients": None,
        }

    return None


def collect(results_dir: str) -> list[dict[str, Any]]:
    rows = []
    skipped = []

    for run_dir in sorted(Path(results_dir).iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            payload = json.load(f)

        row = classify_and_extract(payload["run_id"], payload["metrics"])
        if row is None:
            skipped.append(payload["run_id"])
            continue

        row["git_commit"] = payload["provenance"].get("git_commit", "")[:8]
        row["timestamp_utc"] = payload["provenance"].get("timestamp_utc", "")
        rows.append(row)

    if skipped:
        print(f"Note: {len(skipped)} results/*/metrics.json did not match a "
              f"known shape and were skipped: {skipped}")

    return rows


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No results found yet."

    lines = [
        "| run_id | kind | val F1 macro | test F1 macro (95% CI) | n test patients | commit |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        val = f"{r['val_f1_macro']:.4f}" if r["val_f1_macro"] is not None else ""
        if r["f1_macro_point"] is not None:
            test = f"{r['f1_macro_point']:.4f} [{r['f1_macro_ci_lower']:.4f}, {r['f1_macro_ci_upper']:.4f}]"
        else:
            test = ""
        n_patients = str(r["n_test_patients"]) if r["n_test_patients"] is not None else ""
        lines.append(f"| {r['run_id']} | {r['kind']} | {val} | {test} | {n_patients} | {r['git_commit']} |")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out", default=None, help="Optional path to also write the table to a file")
    args = parser.parse_args(argv)

    rows = collect(args.results_dir)

    if not rows:
        print(f"No results found under {args.results_dir}/ yet. "
              f"Nothing has finished training or been evaluated.")
        return

    table = format_markdown_table(rows)
    print(f"\nFound {len(rows)} result(s):\n")
    print(table)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"# Results summary\n\nGenerated from {len(rows)} result(s) actually present in "
                     f"{args.results_dir}/. Anything not listed here has not run yet.\n\n")
            f.write(table + "\n")
        print(f"\nAlso written to {args.out}")


if __name__ == "__main__":
    main()
