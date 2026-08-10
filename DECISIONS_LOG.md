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
