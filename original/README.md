# Code Supplement for NSCF

This archive accompanies **Neural Supervised Clustering for Learning
Individualized Treatment Rules with Many Treatments**.

It contains the implementation used for the current paper, exact paper-scale
job templates, seed-level simulation results, real-data summary results, and
the scripts used for uncertainty analysis and figure generation.

## Directory map

- `src/its_nn_value_first_grouped/`: NSCF neural representation, fusion path,
  alternating optimization, grouped refitting, and model selection.
- `src/its_compare_value_first_2_grouped/`: synthetic data generation, NSCF
  runner, Neural Q, ridge, lasso, elastic net, random forest, and SVR.
- `src/its_nn_value_first_realdata/`: PDX loading, grouped splitting,
  propensity estimation, Hájek value evaluation, and real-data policy wrapper.
- `src/its_compare_value_first_realdata/`: training-only feature screening,
  treatment-main-effect residualization, grouped-CV comparison runner, and
  full-data descriptive refit.
- `external_baselines/scaf/`: the aligned R implementation used to obtain the
  SCAF comparison rows.
- `slurm/`: complete 50-replication simulation, 50-by-5 PDX grouped-CV,
  repeated-validation lambda-selection, and full-data-refit templates.
- `scripts/`: aggregation, uncertainty-analysis, and paper-figure scripts.
- `results/simulation/`: seed-level input data, regenerated uncertainty
  summaries, and the optimization-budget summary used in the supplement.
- `results/realdata/`: 50-repeat held-out values, 250-fold co-clustering
  frequencies, and full-data lambda/path summaries.
- `DATA.md`: PDX data requirements and leakage-prevention details.
- `EXPERIMENTS.md`: exact experiment-to-code and figure mapping.

The first code-supplement draft used shortened `nscf_*` package names. Those
stale duplicates have been removed so that the archive exposes one canonical
`src/its_*` implementation.

## Python environment

The reported experiments used Python 3.10. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The PDX `.rda` loader also requires R and `pyreadr`. The aligned SCAF runner
requires R packages `jsonlite`, `Rcpp`, `glmnet`, `grpreg`, and
`randomForest`.

Add the source tree to the module path:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## Quick validation

Run the small synthetic smoke test:

```bash
bash scripts/run_smoke_test.sh
```

The smoke test checks imports and the end-to-end command-line workflow; it is
not intended to reproduce paper-scale results.

## Reproduce the paper simulations

The complete paper-facing grid uses 50 seeds per cell:

```bash
sbatch slurm/nscf_paper_simulations_50reps.sbatch
```

Cluster account, partition, modules, and concurrency limits may be supplied by
the local environment. Activate the Python environment before submission. The
script itself contains the exact statistical and optimization settings used in
the paper.

The aligned SCAF cells can be reproduced separately:

```bash
bash external_baselines/scaf/aligned/run_scaf_paper_cells_50.sh
```

The included seed-level merged file is the direct input to the paper plotting
and uncertainty scripts.

## Reproduce the PDX analysis

Set the processed data path and submit all 250 outer-fold jobs:

```bash
export NSCF_PDX_DATA="/path/to/split.cm.data.rda"
sbatch slurm/nscf_pdx_grouped_cv_50x5.sbatch
```

Every repeat partitions the 32 PDX models into five disjoint outer test folds.
Feature screening, scaling, treatment-main-effect estimation, propensity
estimation, tuning, and partition selection are repeated without the outer test
fold.

For the descriptive full-data lambda/path analysis:

```bash
sbatch slurm/nscf_pdx_full_data_lambda_selection_20.sbatch
sbatch --dependency=afterok:<JOB_ID> slurm/nscf_pdx_full_data_refit.sbatch
```

Held-out policy values do not enter the full-data lambda selection.

## Regenerate summaries and figures

The archive contains the machine-readable inputs needed for these commands:

```bash
python scripts/analyze_simulation_inference.py
python scripts/plot_paper_regret_figures.py
python scripts/plot_paper_realdata_final_figures.py
```

The machine-readable values in Supplementary Table S2 are in
`results/simulation/optimization_budget_summary.csv`; the corresponding
paper-scale job template is
`slurm/nscf_optimization_budget_sim_3x5.sbatch`.

To aggregate newly generated 250-fold PDX run directories first:

```bash
export NSCF_REALDATA_RUNS="$PWD/runs/realdata_grouped_cv"
python scripts/plot_paper_cm_balanced_cv_50repeats.py
```

Output locations can be overridden with the environment variables documented
at the top of each script.

## Legacy internal names

Some output rows use `value_first_nn`, the historical implementation identifier
for NSCF. It denotes the same validation-selected NSCF procedure reported in
the manuscript; it is not a separate prediction-first method.
