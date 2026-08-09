# Reproducibility checklist

This checklist is built from two sources. First, the specific failures found in the
review of the previous version of this project (listed below with the rule that now
prevents each one). Second, the general practice checklists used across the field:
the ML reproducibility checklist associated with Pineau et al. 2021, and CLAIM
(Checklist for Artificial Intelligence in Medical Imaging), 2024 update, which is the
reporting standard for this paper's target venue and domain.

## Failures found in the previous version, and the rule that now prevents each one

1. Failure: the final headline result and an earlier ablation row were computed by
   two different loaders that both claimed, only in a code comment, to be "the Malawi
   test set."
   Rule: src/fetal_ai/data/splits.py is the only place any split is defined. Every
   script and notebook imports from it. There is no second path.

2. Failure: a module level default (CLASSES = CLASSES_4C) silently leaked a 4th class
   into a 3 class experiment's plot legend, because a fallback branch relied on it.
   Rule: src/fetal_ai/utils/config.py raises an error on any missing required field.
   Nothing in this codebase has a silent default for anything that changes what data
   or classes are used. If a value matters, it must be written explicitly in the
   config file that produced the result.

3. Failure: the pre-built data folder was downloaded as an opaque zip from Google
   Drive, with no script in the repo showing how its train/val/test split was made,
   and no way to verify it was patient disjoint.
   Rule: data is fetched only from its original, cited, DOI backed source
   (scripts/01_fetch_data.py), and every split is built from a checksummed manifest
   that records patient IDs (scripts/03_build_splits.py). tests/test_no_leakage.py
   checks patient disjointness automatically, on every push, before training starts.

4. Failure: Table 1 in the paper mixed numbers from two different, unrelated
   experiments (raw per country counts and a separate per country train/val split
   from a different sweep), producing numbers that did not add up.
   Rule: results/RESULTS_LEDGER.md maps exactly one run_id to exactly one paper
   table or figure. A number never appears in the paper unless it has a ledger entry
   pointing at the exact script, config, and commit that produced it.

5. Failure: the headline number (98.7 percent F1 on n=75) was reported without a
   confidence interval, without multiple seeds, and without a same protocol baseline
   (pooled fine tuning without LOCO or model soup) to compare against.
   Rule: src/fetal_ai/evaluation/bootstrap.py computes patient level bootstrap
   confidence intervals (never image level, since images from the same patient are
   correlated and bootstrapping at the image level understates the true uncertainty).
   Every reported number in the ledger includes a confidence interval, multiple
   random seeds, and, where relevant, a same protocol baseline.

## General checklist (based on Pineau et al. 2021 and CLAIM 2024)

Models and algorithms
- [ ] A clear description of the model architecture, with a citation for any pretrained
      weights and the exact source of those weights.
- [ ] All hyperparameters and how they were chosen (grid, prior work, or fixed budget).
- [ ] The exact number of parameters and inference cost, reported once, not estimated.

Theoretical claims
- [ ] Every claim in the paper ("X percent points improvement", "clinically correct
      attention", "unseen country generalizes") is backed by a specific reported number
      with a confidence interval, or is explicitly marked as a qualitative observation.

Data (this is where CLAIM 2024 overlaps most directly)
- [ ] Data source, version, and access date, with a permanent identifier (DOI).
- [ ] Inclusion and exclusion criteria for images and patients.
- [ ] Exact patient counts and image counts per class, per country, per split, as a
      table generated directly from the manifest, not typed by hand.
- [ ] Confirmation of patient level split disjointness, with the test that proves it.
- [ ] Description of any preprocessing, and whether it was fit only on training data.
- [ ] Ground truth definition and how labels were produced (who labeled, how).

Code
- [ ] A public repository at a tagged commit for every reported result.
- [ ] A requirements file with pinned versions.
- [ ] Instructions that a stranger could follow start to finish, with no undocumented
      manual steps.
- [ ] Automated tests that catch the two most damaging classes of bugs for this kind
      of project: data leakage and silent misconfiguration.

Experiments
- [ ] Multiple random seeds reported, with mean and spread, for any result used to
      make a comparative claim.
- [ ] Confidence intervals or another measure of uncertainty for every headline number.
- [ ] A same protocol baseline for every proposed component (this is what closes the
      "missing pooled fine tuning baseline" and "missing ensembling baseline" reviewer
      concerns).
- [ ] Compute used, reported honestly (GPU type, approximate hours).

## What "traced" means in practice for this project

Every result file (results/<run_id>/metrics.json) embeds:
- git_commit: the exact commit hash of the code that produced it
- config_hash: a hash of the exact YAML config used
- data_manifest_hash: a hash of the exact manifest file used, which itself contains
  per file checksums and the patient to split assignment
- seed: the random seed used
- timestamp and environment: python and package versions at run time

This means any number can be traced backward from paper to run_id to exact code,
exact config, and exact data, with no step that depends on memory or a comment.
