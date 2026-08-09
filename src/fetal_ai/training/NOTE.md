Training loop code goes here. Deliberately not written yet, for the same
reason as src/fetal_ai/models/NOTE.md: it depends on model code and real
data shapes that have not been verified yet.

Built in Phase 4. See README.md "Order of operations."

When this does get built, it will call:
  - fetal_ai.utils.seed.set_seed()             at the very start
  - fetal_ai.utils.config.load_config()          to load and validate the config
  - fetal_ai.data.splits.load_split()            to get the exact patient split
  - fetal_ai.utils.tracking.start_run()           to log to Weights and Biases
  - fetal_ai.provenance.build_provenance_stamp()  before saving any result
  - fetal_ai.provenance.save_run_result()         to write metrics.json

in that order, so a training run is traceable from the first line.
