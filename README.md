# Fetal Ultrasound Standard Plane Classification for Low Resource African Settings

This repository is a ground up rebuild of the AfroFetalNet project. It exists to fix a
specific problem found during review: results that could not be traced back to a single,
auditable pipeline. Every rule in this repo exists to close one gap that caused a real
problem in the previous version. See REPRODUCIBILITY.md for the full list.

## The one rule that matters most

There is exactly one way to load any given split of data, and exactly one function that
evaluates a model on any given split. If two experiments claim to use "the same test set",
they must call the same function with the same config, not just share a comment saying so.
Nothing gets loaded from a hand built folder that does not have a script proving how it was
built.

## How to trace any number in the paper back to its source

1. Open results/RESULTS_LEDGER.md
2. Find the paper table or figure you are checking
3. It lists a run_id
4. Open results/<run_id>/metrics.json
5. That file contains the exact git commit, the exact config file (hashed and embedded),
   and the exact data manifest hash used to produce it
6. Re-run scripts/05_train.py or scripts/06_evaluate.py with that same config to reproduce it

## Project layout

```
configs/            YAML configs. Every field is required, nothing defaults silently.
data/                Raw data is never committed. manifest/ holds checksummed file lists
                      and patient level split assignments, and those ARE committed.
src/fetal_ai/        All real logic lives here. Notebooks and scripts only call into this.
scripts/             Numbered, run in order. Each one is a single traceable step.
tests/               Leakage tests, split determinism tests, config validation tests.
                      These run in CI on every push, before any GPU time is spent.
notebooks/           Thin orchestration for Kaggle. No data loading logic lives here.
results/             One folder per run, plus the ledger that maps paper claims to runs.
```

## Order of operations

This project is built in phases, and each phase is verified before the next one starts.

Phase 0: environment and tooling (this commit)
Phase 1: fetch raw data from its original public source, build a checksummed manifest
Phase 2: build patient level splits (LOCO folds, pooled baseline, held out country rotation)
Phase 3: run and pass the leakage and determinism tests, on the real manifest, before any
          model is trained
Phase 4: model and training code
Phase 5: evaluation, bootstrap confidence intervals, statistical comparisons
Phase 6: explainability
Phase 7: write the camera ready paper directly from results/RESULTS_LEDGER.md

We do not move to a phase until the previous one is green.

## Data sources (fixed, public, cited)

- FETAL_PLANES_DB (Burgos-Artizzu et al. 2020), Zenodo DOI 10.5281/zenodo.3904280
- African multi-centre fetal ultrasound dataset (Sendra-Balcells et al. 2023),
  Zenodo DOI 10.5281/zenodo.7540448

Both are fetched by scripts/01_fetch_data.py directly from Zenodo, never from a pre-zipped
mirror, and every downloaded file is checksum verified against the manifest before use.

## Tooling choices and why

- Git plus GitHub for code. Tag a commit at every reported result, not just at releases.
- Weights and Biases for experiment tracking. Free for academic use, integrates with
  Kaggle secrets, and every run automatically logs the git commit hash, so a W&B run
  and a git commit always point at each other.
- Plain YAML configs with a strict loader (src/fetal_ai/utils/config.py) that raises an
  error on any missing required field, instead of Hydra. Hydra is powerful but the failure
  mode we are protecting against is silent defaults, and a strict loader closes that gap
  with less machinery to learn. Hydra multirun remains a reasonable upgrade later if the
  number of experiment combinations grows.
- pytest plus GitHub Actions for automated checks that run on every push, free of charge,
  and do not require a GPU.
- A checksummed CSV manifest instead of DVC. Both raw datasets already have permanent,
  citable DOIs on Zenodo, so the canonical copy of the data is the DOI itself. DVC adds
  real value when data lives nowhere else and needs its own remote storage, which is not
  our situation. A manifest with SHA256 checksums per file, committed to git, gives the
  same guarantee (anyone can verify they have the exact same bytes) with far less setup,
  and it works cleanly inside ephemeral Kaggle sessions.
- Kaggle notebooks for GPU compute, but strictly as thin orchestration: clone the repo at
  a pinned commit, install requirements, call into scripts/, and log to Weights and Biases.
  No dataset loading logic, no class name lists, no split logic is ever written directly
  inside a notebook cell.
