#!/bin/bash
# Runs the full data pipeline, in order, stopping immediately if any step
# fails (set -e). This script exists as a literal, readable record of the
# exact commands, in the exact order, that turn raw Zenodo downloads into
# a verified, leakage free manifest and split set. It intentionally does
# not include training, training is a separate, much longer step that
# should be run and tracked one experiment at a time, not looped through
# blindly.
#
# Usage:
#   bash scripts/run_data_pipeline.sh

set -e

echo "Step 0: environment check"
python scripts/00_check_environment.py

echo ""
echo "Step 1: fetch raw data from Zenodo"
python scripts/01_fetch_data.py

echo ""
echo "Step 2: build the standardized manifest"
python scripts/02_build_manifest.py

echo ""
echo "Step 3: build every split"
python scripts/03_build_splits.py

echo ""
echo "Step 4: verify no leakage, against the real data"
python scripts/04_verify_no_leakage.py

echo ""
echo "Data pipeline complete. Commit data/manifest/ to git now, before"
echo "training anything, so the exact splits used are pinned in history."
