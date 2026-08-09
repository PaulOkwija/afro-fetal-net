# Results ledger

This file is the single source of truth connecting the paper to the code
that produced it. A number does not get written into the camera ready
paper unless it has a row here. Fill in a row before running the
experiment, not after, so the intended purpose of a run is recorded even
if the run itself fails or gets rerun later.

Status values: planned, running, done, superseded

| Paper location | What it reports | Config used | run_id | W&B link | Status | Notes |
|---|---|---|---|---|---|---|
| Table 1 | Dataset counts, raw and per split | configs/data.yaml + scripts/02_build_manifest.py | (generated, not a training run) | n/a | planned | generated directly from the manifest, never typed by hand |
| Table 3 | LOCO per fold validation results | configs/experiment/loco_africa.yaml | | | planned | |
| Table 4 row 1 (zero shot) | Spain model evaluated directly on Malawi, no adaptation | configs/experiment/baseline_spain.yaml (eval on Malawi held out set) | | | planned | |
| Table 4 row 2 (+CLAHE) | Same as above, with CLAHE preprocessing on | same config, preprocessing.use_clahe: true | | | planned | |
| Table 4 row 3 (+LOCO) | Mean of LOCO fold checkpoints evaluated on Malawi | configs/experiment/loco_africa.yaml | | | planned | must use the same held out loader as row 4, see REPRODUCIBILITY.md item 1 |
| Table 4 row 4 (+Model Soup) | Averaged LOCO checkpoints evaluated on Malawi | configs/experiment/loco_africa.yaml + averaging step | | | planned | must use the exact same held out loader as row 3, this is the row that broke last time |
| New: pooled baseline | Same protocol, no LOCO, no soup | configs/experiment/pooled_baseline.yaml | | | planned | requested by both reviewers, did not exist before |
| New: country rotation | Each country held out in turn | configs/experiment/country_rotation.yaml | | | planned | answers "does unseen beat validation for every country, or just Malawi" |
| Table 5 | Grad-CAM attention metrics on Malawi | configs/experiment/loco_africa.yaml + explainability step | | | planned | |
| Section 4.5 | Final model per class F1, confusion matrix, CI | configs/experiment/loco_africa.yaml + bootstrap step | | | planned | must include bootstrap CI and multiple seeds, not a single point estimate |
| Figure 1 | CLAHE preprocessing example | n/a, visualization only | n/a | n/a | planned | |
| Figure 3 | t-SNE domain shift | configs/experiment/baseline_spain.yaml, features extracted | | | planned | legend classes must equal configs classes exactly, no 4th class, see REPRODUCIBILITY.md item 2 |
| Figure 4 | Grad-CAM visualizations | configs/experiment/loco_africa.yaml | | | planned | |

## How to add a row

1. Before running anything, add a row with Status = planned, naming the
   exact config file you intend to use.
2. Run the script. It will print a run_id and, if it succeeds, write
   results/<run_id>/metrics.json.
3. Update the row: paste the run_id, the Weights and Biases link, and
   change Status to done.
4. If you ever rerun something that changes the number (different seed,
   fixed bug, more epochs), do not overwrite the row. Add a new row,
   mark the old one's Status as superseded, and note why in Notes. The
   paper should always cite the current row, but the superseded ones
   stay in git history as part of the audit trail.
