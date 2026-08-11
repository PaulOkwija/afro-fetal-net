# Data directory

Nothing under data/raw/ or data/interim/ is committed to git (see
.gitignore). Raw image files are large, and more importantly, committing
them would create a second, uncontrolled copy of "what the data is" that
could drift from the manifest. The manifest is the copy of record.

data/manifest/ does not need to be committed to git for reproducibility,
and is not required to be. It contains generated data (not code):

- manifest.csv: the standardized, checksummed table built by
  scripts/02_build_manifest.py, see src/fetal_ai/data/manifest.py for the
  column definitions.
- splits/*.json: every split this project uses, built by
  scripts/03_build_splits.py, see src/fetal_ai/data/splits.py.

Reproducibility of this directory does not depend on git history: every
raw file is checksum verified against Zenodo before use (fetch.py), and
manifest.py and splits.py are deterministic given the same raw data and
a fixed seed. A run's exact manifest content is captured independently
by data_manifest_hash in that run's results/<run_id>/metrics.json, see
src/fetal_ai/provenance.py. This is why get_git_commit ignores
data/manifest/ when checking for uncommitted changes, rebuilding it
fresh every Kaggle session is expected and fine, no commit needed.

Committing data/manifest/ anyway is still reasonable if you want a
convenient, browsable history of exactly what the manifest looked like
at different points, but it is a convenience, not a requirement.

To reconstruct data/raw/ and data/manifest/ from nothing but this
repository:

    python scripts/00_check_environment.py
    python scripts/01_fetch_data.py
    python scripts/02_build_manifest.py
    python scripts/03_build_splits.py
    python scripts/04_verify_no_leakage.py

The last step re-checksums every file against the manifest it just
built and independently re-verifies every split for patient level
leakage. If anything about the source data changed, or a config mapping
is wrong, that will be reported loudly rather than silently proceeding
with a manifest or split that no longer matches what it should.
