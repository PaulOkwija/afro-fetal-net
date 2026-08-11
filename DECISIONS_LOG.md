# Decisions log

This is a running, dated record of real findings and the reasoning behind
each decision made because of them. It is not a replacement for git
commit messages, commit messages say what changed, this says why, in the
order it was actually figured out. Two uses for this file: remembering
why the pipeline looks the way it does, and drafting material for the
paper's data quality and limitations sections, which need to describe
exactly this kind of thing under CLAIM 2024.

Write an entry whenever a number looks surprising, an assumption turns
out wrong, or a decision gets made that isn't obvious from the code
alone. Paste real output, not a paraphrase of it, the specific numbers
are what make an entry useful later.

---

## 2026-08-09 Kaggle kernel numpy/pandas ABI mismatch

**Trigger:** `import pandas` failed inside the Kaggle notebook kernel
with `ValueError: numpy.dtype size changed, may indicate binary
incompatibility. Expected 96 from C header, got 88 from PyObject`,
right after `pip install -r requirements.txt` had run cleanly moments
before.

**Evidence:** the pip install step itself had printed a long list of
dependency warnings (`jaxlib requires numpy>=2.0, but you have numpy
1.26.4`, and similar for a dozen other Kaggle-preinstalled packages).
Meanwhile `!python scripts/00_check_environment.py`, run as a subprocess
right after the same install, worked fine.

**Finding:** Kaggle's base image ships numpy 2.x, and most of its
preinstalled packages are compiled against that ABI. `requirements.txt`
forced numpy down to 1.26.4, which pip does on disk but does not undo
for the packages already compiled against 2.x. The result is a
genuinely inconsistent install, not a stale kernel memory issue.
Restarting the kernel only ever worked by accident, depending on import
order.

**Decision:** stop fighting Kaggle's own numpy version. Two changes:
loosen the numpy-family pins in `requirements.txt` to versions
compatible with numpy 2.x instead of forcing a downgrade, and treat
every script that touches numpy-dependent libraries as something that
runs via `!python scripts/...` (a fresh subprocess), never as an inline
import in the long-lived notebook kernel. The second part is really
just applying a rule the project already had for its numbered scripts,
consistently, to ad hoc inspection work too.

**Files touched:** requirements.txt (pending), notebooks/00_kaggle_setup.ipynb

**Open question:** confirm the loosened numpy pin doesn't itself break
anything once scikit-learn and other pinned versions are checked
against it.

---

## 2026-08-09 manifest.py committed to the wrong directory

**Trigger:** `TypeError: build_manifest() got an unexpected keyword
argument 'csv_separator'` when running `scripts/02_build_manifest.py`,
even though the calling script had already been updated to pass that
argument.

**Evidence:**
```
commit 6ec8046...
 configs/data.yaml            |  65 +++++++------
 data/manifest/manifest.py    | 218 +++++++++++++++++++++++++++++++++++++++++++
 requirements.txt             |   4 +-
 scripts/02_build_manifest.py |   2 +
```

**Finding:** the updated file landed at `data/manifest/manifest.py`, a
brand new file, instead of overwriting the real one at
`src/fetal_ai/data/manifest.py`. Two folders both contain "manifest" in
the name, one is generated output (`data/manifest/`), one is source code
(`src/fetal_ai/data/`), and it's an easy mix-up.

**Decision:** delete the wrongly placed file, replace the real one.
No code or design change, purely a location mistake. Noted here anyway
because it cost a full round trip to diagnose, and the fix (checking
`git log -1 -- <path>` for the exact file in question, not just eyeballing
`git show --stat`) is a useful habit worth remembering.

**Files touched:** deleted data/manifest/manifest.py, replaced
src/fetal_ai/data/manifest.py

**Open question:** none, resolved.

---

## 2026-08-09 Raw data structure differs from the placeholder config assumptions

**Trigger:** planned inspection before running scripts/02_build_manifest.py
for real, per REPRODUCIBILITY.md's rule against guessing column mappings.

**Evidence:**
```
African columns: ['Patient_num', 'Plane', 'Train', 'Center', 'Filename']
Malawi folder contents: ['patient019_MWI_plane3.png', ...]
FETAL_PLANES_DB columns: ['Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train ']
```

**Finding:** two real structural differences from the placeholder config.
The African dataset's images live in per-country subfolders
(`Zenodo_dataset/<Center>/<filename>.png`), not one flat folder, and its
CSV's `Filename` column has no extension and no country prefix.
FETAL_PLANES_DB's metadata file is semicolon delimited, not comma
delimited, which is why an unspecified `pd.read_csv` merged the whole
header into a single column name. Also, the label strings are `Fetal
abdomen`, `Fetal brain`, `Fetal femur`, capital F, while the placeholder
config had them lowercase, which would have silently dropped every
African row (manifest.py matches label_mapping keys case sensitively,
on purpose).

**Decision:** added `csv_separator` and `group_subdir` parameters to
`build_manifest`, so the per-row image path can be built from the group
column when needed, and the CSV reader can be told the real delimiter.
Updated configs/data.yaml with the confirmed real column names, real
paths, and correct label casing.

**Files touched:** src/fetal_ai/data/manifest.py, configs/data.yaml,
scripts/02_build_manifest.py

**Open question:** none, confirmed working against synthetic data
mirroring the real structure before handing back.

**Follow-up 2026-08-09:** scripts/04_verify_no_leakage.py was written
before this fix and never updated, its checksum re-verification still
defaulted to the old flat `data/raw/african_multicentre/Images` path.
Running it after the manifest and split fixes reported 340 "missing
file" problems, every single one under that stale path. The LOCO,
pooled baseline, and country rotation checks in the same run all
reported 0 problems, which was the tell that this was a leftover
assumption in one script, not a real data or split issue, since those
checks work entirely from the manifest and never touch the filesystem
directly. Fixed the default path and added a group_subdir parameter to
verify_checksums, mirroring the same fix already made in manifest.py.
Confirmed against synthetic data both that the fix resolves it and that
the unfixed version genuinely reproduces the original failure, so the
test isn't just passing by coincidence.

**Files touched:** scripts/04_verify_no_leakage.py

---

## 2026-08-09 CLAHE contrast score definition is not specified in the original paper

**Trigger:** writing src/fetal_ai/data/dataset.py, needed a concrete
formula for the paper's "global contrast score" used to decide which
images get selective CLAHE applied.

**Evidence:** the paper states CLAHE is applied "selectively to images
with a global contrast score below 35 (the lower quartile of the
African dataset)" but never defines the formula for that score.

**Finding:** no way to confirm the original authors' exact definition
without their source code, which isn't available. This is a genuine
gap, not something to quietly paper over with an assumption presented
as fact.

**Decision:** defined contrast score as RMS contrast, the standard
deviation of grayscale pixel intensity, a standard and common
definition, and said so explicitly in the module docstring rather than
implying it matches the original implementation. Confirmed the function
actually distinguishes a synthetic low contrast image from a high
contrast one, and that the resulting selective CLAHE only modifies the
image below threshold, before relying on it for anything.

**Files touched:** src/fetal_ai/data/dataset.py

**Open question:** worth a specific sentence in the camera-ready
methods section stating this is our operational definition, since the
original paper's phrasing was ambiguous and a reviewer could reasonably
ask. If the original authors' code or a clearer definition ever
surfaces, this is the one function that needs to change.

---

## 2026-08-09 Patient_num collides across African countries, and real duplicate images exist in both source datasets

**Trigger:** `02_build_manifest.py` reported only 61 unique patients for
the African dataset, when 5 countries times up to 25 patients each
should be close to 125.

**Evidence:**
```
Unique Patient_num values IGNORING country: 64
Unique patient identities WITH country prefix: 127
```
and, from a dedicated diagnostic script comparing checksums row by row:
```
african_multicentre: same file referenced twice in metadata: 0
                      genuinely different files, identical bytes: 70
fetal_planes_db:      same file referenced twice in metadata: 0
                      genuinely different files, identical bytes: 192
```

**Finding:** two separate real issues, not code bugs.

First, `Patient_num` in the African dataset's raw metadata is not
globally unique, it's assigned independently per country, so
`Patient_num=17` in Malawi and `Patient_num=17` in Egypt are two
different real people sharing a string. Our own `patient_id` field
(just `str(Patient_num)`, no country attached) was silently merging
them.

Second, the "duplicate" images are real, not an artifact of our
pipeline. In the African dataset, specific patient number pairs within
the same country (for example Malawi 17 and 26, or Malawi 19 and 8)
have byte-identical images across all three planes, abdomen, brain, and
femur. That's the signature of the same real patient entered twice
under two different patient numbers in the source dataset itself. In
FETAL_PLANES_DB, duplicates are almost always within the same patient,
consecutive video frame numbers (`Plane3_2_of_3` vs `Plane3_1_of_3`),
consistent with a sonographer holding the probe still for a moment
during acquisition, a more benign explanation. One exception is worth
its own note: patient 1272 has a byte-identical abdomen image and brain
image, which crosses a class boundary and looks like a genuine labeling
or export error in the source data rather than a static frame.

**Decision:** prefix `patient_id` with the group (country, or "spain")
so it is globally unique, closing the collision. Kept the existing
checksum based deduplication (it was already doing the right thing
mechanically), but added a `duplicate_report_path` so every dropped row
is written to an audit CSV alongside which row it duplicated, instead of
just a printed count. This turns a previously silent decision into a
citable one, and the audit files are exactly the source material for a
paragraph in the paper's data section describing this as a known
characteristic of both public datasets.

**Files touched:** src/fetal_ai/data/manifest.py, scripts/02_build_manifest.py

**Open question:** once the manifest is rebuilt with these fixes, check
whether the corrected unique patient counts and the per-country
duplicate concentration (Malawi and Algeria lost the most images to
deduplication) still look the way they did in the investigation, and
decide whether that concentration is worth a specific sentence in the
paper beyond the general note about duplicate entries.

**Resolved 2026-08-09:** reran scripts/02_build_manifest.py against the
fixed manifest.py. African unique patients went from 61 to 116, close
to the 127 predicted in Investigation 1, the remaining gap is patients
whose entire image set was a full duplicate of another patient and
correctly collapsed to one surviving identity. Combined total across
both datasets is 1243 unique patients, and 1127 (fetal_planes_db) + 116
(african_multicentre) = 1243 exactly, a clean check that there is no
further collision between the two datasets. Image counts per country
are unchanged from before the fix (Algeria 57, Egypt 73, Ghana 75,
Malawi 60, Uganda 75), which is expected, deduplication runs on
file_sha256 and was never affected by the patient_id fix, only the
patient identity each surviving image maps to changed.

The concentration question is answered: Malawi and Algeria are still
the two countries that lost the most images to deduplication (Malawi
75 to 60, a 20 percent reduction; Algeria 75 to 57, a 24 percent
reduction), same as in the original investigation. This is worth a
specific, plainly stated sentence in the paper's dataset section, not
folded into a generic deduplication note: the corrected, deduplicated
Malawi held-out test set is n=60 (20 abdomen, 21 brain, 19 femur), not
the n=75 reported in the original submission. This makes the already
small held-out set smaller, which strengthens rather than weakens the
case for bootstrap confidence intervals on every number computed from
it, and it should be stated as a correction, not buried.

**Independently verified 2026-08-09:** before reporting the 60 figure
anywhere, re-checked the claim using a tool with no relationship to our
own pipeline, in case the sha256 computation in manifest.py had a bug.
Ran `md5sum` (standard Unix tool, different hash algorithm entirely)
directly against the actual downloaded files for the Malawi patient
017 vs 026 pair:
```
27ae27cab0fff14a80b38927533f80f3  Malawi/patient017_MWI_plane0.png
27ae27cab0fff14a80b38927533f80f3  Malawi/patient026_MWI_plane0.png
```
Identical. Two independent hash algorithms (our SHA256, this MD5)
agreeing that two files are byte-identical is conclusive, not
coincidental. Also directly reconfirmed the raw CSV itself: Malawi has
100 rows total, 75 once filtered to the 3-class task, matching the
paper's original number exactly, so the 75 to 60 reduction is entirely
attributable to the deduplication step, not a metadata reading error
anywhere upstream. Decision stands: report both numbers in the paper,
75 as the source metadata's nominal count, 60 as the verified count of
distinct images, with the duplicate finding stated as a real
characteristic of the public dataset, not a discrepancy to reconcile
away.

---

## 2026-08-10 pr_auc_macro was computed with roc_auc_score, a naming bug

**Trigger:** a review pass flagged that src/fetal_ai/evaluation/metrics.py
stored a roc_auc_score result under the key pr_auc_macro, and asked
whether that was intentional before touching anything.

**Evidence:** every configs/experiment/*.yaml already listed
"pr_auc_macro" in evaluation.metrics, and neither the original paper
nor the AFRICAI review mentions ROC AUC or PR AUC by name anywhere.

**Finding:** genuine naming bug from when metrics.py was first
scaffolded, not an intentional metric choice. Given the small,
imbalanced held out sets this project works with (Malawi's 60 patients),
PR AUC (average precision) is the more informative metric anyway, it is
more sensitive to minority class precision than ROC AUC, which treats
false positives and false negatives symmetrically.

**Decision:** compute pr_auc_macro correctly with
average_precision_score, and keep roc_auc_macro alongside as a
separate, correctly named metric. First attempt at the fix assumed
average_precision_score accepts multiclass integer y_true directly the
same way roc_auc_score's multi_class="ovr" does, that assumption was
wrong. average_precision_score has no labels parameter, so a batch
missing one class (a real case on small bootstrap resamples) produced a
shape error rather than a graceful result. Caught this by testing the
missing class case specifically, not just the case where everything
works. Fixed by explicitly binarizing y_true with label_binarize
against the fixed class list before calling average_precision_score,
which also turned out to make the missing class case handled better
than roc_auc_score's own behavior (a real number, 0.36, instead of
roc_auc_score's nan). Confirmed against 20 simulated bootstrap style
resamples, matching how evaluation/bootstrap.py actually calls this
function repeatedly.

**Files touched:** src/fetal_ai/evaluation/metrics.py

**Open question:** none, both metrics confirmed correct and
distinguishable from each other, and the missing class edge case
confirmed handled for both before calling this done.

**Note on this entry itself:** the commit that shipped this fix
(4f1b8d5, "Made some adjustments to the metrics") only touched
metrics.py, this log entry was written separately afterward once a
review pass noticed the log and the code had drifted apart. Worth
remembering going forward: commit the log entry in the same commit as
the fix it describes, not as an afterthought, otherwise this file stops
doing the one thing it exists for.

---

## 2026-08-10 requirements.txt numpy/pandas left fully unpinned

**Trigger:** review pass on the same commit, checking whether the
Kaggle numpy ABI fix from an earlier entry had actually been resolved
correctly in requirements.txt.

**Evidence:** requirements.txt had `numpy` and `pandas` with no version
constraint at all, versus every other dependency in the file being
pinned to an exact version, and the file's own header comment saying
"Pin exact versions... do not let Kaggle's preinstalled versions
silently differ from what is written here."

**Finding:** this does resolve the original ABI conflict (nothing forces
a downgrade against Kaggle's numpy 2.x base image anymore), but it
contradicts the file's stated purpose. Fully unpinned means a Kaggle
session run months from now could silently resolve a different numpy
than whatever this project was actually tested against, which is
exactly the kind of drift this file exists to prevent.

**Decision:** pinned numpy to 2.0.2 and pandas to 2.2.3, both numpy 2.x
releases, compatible with Kaggle's base image without forcing a
downgrade, and confirmed compatible with the other pinned versions
(scikit-learn 1.5.1, torch 2.4.1) by actually installing this exact
combination and rerunning the full test suite and the metrics.py fix's
own sanity check against it, not just asserting compatibility.

**Files touched:** requirements.txt

**Open question:** if a future Kaggle base image ships a numpy version
outside the 2.x line, this pin will need revisiting, the same way the
original 1.26.4 pin did. Worth checking scripts/00_check_environment.py's
output the first time this project runs on any new Kaggle image
generation, rather than assuming this pin stays valid indefinitely.

---

## 2026-08-10 scripts/06_train.py, and an honest gap in what could be tested locally

**Trigger:** writing the script that wires config, manifest, split,
dataset, model, trainer, and provenance together into one runnable
command, the last piece of Phase 4.

**Finding:** different experiment configs point at different split
types (patient_level_train_val_test, pooled_baseline, loco,
country_rotation), and a split type determines how many training runs
one config actually produces, one for the first two, one per fold for
LOCO, one per held out country for the rotation. The script dispatches
on split_type rather than having separate scripts per experiment kind,
keeping the single training function (train_model) as the only place
the actual training loop exists, matching the whole point of this
rebuild.

**Testing gap, stated plainly rather than glossed over:** ran a full
integration test end to end, real config, real manifest, real
synthetic images, real dataset and dataloader construction, real
training loop, real checkpoint, real provenance stamp. It passed. But
that test used pretrained=False, because this development sandbox has
no network access to huggingface.co, where timm downloads ImageNet
weights from, and the Spain baseline config uses pretrained=True. The
traceback from the one attempt made with pretrained=True showed the
code correctly attempting the real download and failing only on the
network call itself, not on anything in this project's code, but that
is not the same as having confirmed the download and weight loading
actually completes. That first real run on Kaggle, which has internet
access, is the actual confirmation this still needs.

**Decision:** ship the script as tested, with this gap named explicitly
rather than claimed as fully verified. The first real training run
against Kaggle should be watched for successful ImageNet weight
loading specifically, not just assumed to work because everything else
did.

**Files touched:** scripts/06_train.py

**Open question:** confirm on the first real Kaggle run that
pretrained=True actually downloads and loads ImageNet weights
correctly, then this entry can be marked resolved.

**Resolved 2026-08-10:** first real Kaggle run against
baseline_spain.yaml confirmed it. Output showed
`model.safetensors: 100%|...| 21.4M/21.4M`, the ImageNet weight
download and load completed successfully. The run itself then stopped
at the provenance check instead (data/manifest/ was uncommitted, a
separate, correct refusal, see the next entry), but that happened after
the model was already built and downloaded, so this specific gap is
closed. pretrained=True works as expected on Kaggle.

---

## 2026-08-10 First real Kaggle training attempt correctly refused on uncommitted data/manifest

**Trigger:** first real run of scripts/06_train.py against
baseline_spain.yaml on Kaggle, after the ImageNet weights loaded and
the train/val datasets built (3061 train images from 732 patients, 726
val images from 169 patients, both matching exactly what
scripts/03_build_splits.py reported earlier), the run stopped with
RuntimeError from get_git_commit: "Working tree has uncommitted
changes... data/manifest/".

**Finding:** this is the provenance guard in
src/fetal_ai/provenance.py working exactly as designed, not a bug.
data/manifest/ (manifest.csv, splits/*.json, the two duplicate reports)
had been built in this Kaggle session but never committed to git.
Training against it anyway would mean the checkpoint's provenance stamp
could not actually prove what data it was trained on, since the
manifest sitting on disk had no corresponding git history. This is the
exact failure mode this guard exists to catch, and it caught it before
any GPU time was spent training against unpinned data, not after.

**Decision:** no code change needed. Committed data/manifest/ to git
(git add data/manifest/, commit, push) before rerunning. Worth stating
plainly as a process note: data/README.md already said to commit
data/manifest/ before training, this run is the first real case of that
instruction actually mattering, and the guard caught the gap when the
instruction alone had not been followed yet.

**Files touched:** none, data/manifest/ committed as data, not code

**Open question:** confirm the rerun, after committing data/manifest/,
completes training successfully and produces a real checkpoint and
results/baseline_spain_efficientnet_b0/metrics.json.

**Superseded 2026-08-10:** the decision in this entry, commit
data/manifest/ before every training run, was reasonable in principle
but wrong in practice: it meant a manual git commit on Kaggle every
session, which is exactly the kind of friction that gets skipped under
time pressure, silently reopening the gap this guard exists to close.
See the next entry for the actual fix, narrowing the provenance check
to code changes only, since data/manifest/'s reproducibility never
actually depended on git in the first place.

---

## 2026-08-10 Provenance check no longer requires committing data/manifest/

**Trigger:** direct pushback after the previous entry's fix, do not
want to run git commit on Kaggle every session, the pipeline should
just reproduce the same output every time without that ceremony.

**Finding:** the underlying concern behind the previous entry's fix,
never train against data that cannot be traced back to something,
does not actually require data/manifest/ to be in git. It requires that
the exact data a run used can be identified after the fact.
data_manifest_hash in src/fetal_ai/provenance.py already does exactly
that, computed directly from the manifest file's bytes, regardless of
whether that file is tracked by git. Combined with fetch.py's checksum
verification against Zenodo and manifest.py and splits.py both being
deterministic given a fixed seed, data/manifest/ was already fully
reproducible without a git commit, the commit requirement was ceremony,
not safety.

**Decision:** narrowed get_git_commit's dirty check to exclude
data/manifest/ specifically, using git's pathspec exclude syntax,
tested directly rather than assumed to work. Confirmed three cases:
uncommitted changes limited to data/manifest/ no longer block a run,
an uncommitted change to any actual code file still blocks a run
exactly as before, and the working tree was correctly restored after
testing. Updated data/README.md (which also had a separate real gap,
its reconstruction steps skipped scripts/03_build_splits.py entirely)
and scripts/run_data_pipeline.sh's final message to match, and marked
the previous entry's "commit data/manifest" decision as superseded
rather than leaving two contradictory instructions in this log.

**Files touched:** src/fetal_ai/provenance.py, data/README.md,
scripts/run_data_pipeline.sh

**Open question:** none, all three cases tested directly against the
real git behavior before this was called done.
