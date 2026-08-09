Model architecture code goes here. It is deliberately not written yet.

Writing it now would mean guessing at data shapes (image dimensions after
preprocessing, exact class counts, checkpoint format) before the real
FETAL_PLANES_DB metadata file has actually been downloaded and inspected.
That is the same kind of unverified assumption that caused problems in
the previous version of this project.

This gets built in Phase 4, after scripts/04_verify_no_leakage.py passes
against the real, downloaded data. See README.md "Order of operations."
