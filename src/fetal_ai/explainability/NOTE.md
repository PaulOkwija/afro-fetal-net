Grad-CAM and related explainability code goes here. Deliberately not
written yet, it depends on a trained model existing first.

Built in Phase 6. See README.md "Order of operations."

When built, note the reviewer concern this must address directly:
concentration and entropy of a CAM measure how sharp an attention map
is, not whether it falls on the correct anatomy. If expert annotated
landmarks are available, compare against them directly. If they are not,
say so plainly in the paper rather than describing the result as
"clinically correct attention."
