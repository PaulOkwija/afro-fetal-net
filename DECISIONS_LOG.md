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
