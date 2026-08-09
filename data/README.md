# Data directory

Nothing under data/raw/ or data/interim/ is committed to git (see
.gitignore). Raw image files are large, and more importantly, committing
them would create a second, uncontrolled copy of "what the data is" that
could drift from the manifest. The manifest is the copy of record.

data/manifest/ IS committed to git. It contains:

- manifest.csv: the standardized, checksummed table built by
  scripts/02_build_manifest.py, see src/fetal_ai/data/manifest.py for the
  column definitions.
- splits/*.json: every split this project uses, built by
  scripts/03_build_splits.py, see src/fetal_ai/data/splits.py.

To reconstruct data/raw/ from nothing but this repository:

    python scripts/00_check_environment.py
    python scripts/01_fetch_data.py
    python scripts/02_build_manifest.py
    python scripts/04_verify_no_leakage.py

The last step will compare the freshly built manifest.csv against the
one already committed in git. If they differ, something about the
source data changed, or a config mapping is wrong, and that will be
reported loudly rather than silently proceeding with a manifest that no
longer matches what is on Zenodo.
