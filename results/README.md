# Results directory

Each run writes to results/<run_id>/, and that folder is not committed to
git (see .gitignore), because it can contain large files (checkpoints,
prediction arrays) and because Weights and Biases is the system of record
for run history, not git.

What every results/<run_id>/ folder contains:

- metrics.json: the metrics for that run, plus a full provenance stamp
  (git commit, config hash, data manifest hash, seed). Written by
  src/fetal_ai/provenance.py, and by nothing else.
- checkpoint.pt: the trained model weights, if this run trained a model.
- predictions.csv: per patient, per image predictions, used as the input
  to bootstrap confidence intervals.
- config_used.yaml: a copy of the exact config file used for this run.

RESULTS_LEDGER.md is the one file in this directory that IS committed to
git. It is the map from "this table or figure in the paper" to
"this run_id, which you can look up on Weights and Biases or regenerate
locally." A number does not go into the paper until it has a line in
that ledger.
