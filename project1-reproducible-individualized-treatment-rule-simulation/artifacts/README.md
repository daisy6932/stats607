# Intermediate artifacts

`make reproduce` writes the training log and the original implementation's intermediate diagnostics and runtime configuration here. These files help investigate a run but are not the final deliverables. They are regenerated locally and excluded from version control to avoid committing repetitive logs and temporary plots.

The pipeline starts from synthetic-data generation; it does not need a saved model or intermediate checkpoint. See `../results/` for the committed final figure and table, and `../results/reference/` for the immutable numerical reference used by `make compare`.
