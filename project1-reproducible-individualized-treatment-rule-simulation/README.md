# Project 1 — Reproducible Individualized Treatment Rule Simulation

This project compares three individualized treatment policies: neural supervised clustering (NSCF), a neural Q model, and ridge regression. It reorganizes a selected analysis from the ITR Phase 1 research project into a reproducible pipeline for STATS 607 Unit 1.

## Analysis and data

The simulation generates nonlinear synthetic outcomes with ten treatments in two true groups. It uses seeds 123–125 and 160 training, 80 validation, and 1,000 test observations per repetition. All inputs are generated from `config/experiment.json` and the included source code. No external datasets, patient records, credentials, or access permissions are required.

This intentionally small experiment uses a short training budget. It demonstrates reproducibility, not paper-scale performance. Three repetitions cannot establish general method rankings, and underfitting can affect the neural models. This repository does not reproduce the full paper or PDX real-data experiments.

## Computational environment

System requirements: Python **3.11**, Git, and Make. The tested platform is macOS arm64. Linux and Windows through WSL have not been tested. No GPU, R, cluster account, or notebook server is needed. Internet access is required to install packages; reproduction itself runs offline.

From the cloned `stats607` repository root:

```bash
cd project1-reproducible-individualized-treatment-rule-simulation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

`requirements.txt` pins direct dependencies. `requirements-lock.txt` also pins the transitive dependencies resolved on the tested platform. The environment itself is excluded from version control.

## Reproduce the results

From the project folder, with the environment activated, run:

```bash
make reproduce
```

This command generates the synthetic inputs, trains the three policies, validates the results, and writes the final outputs. It requires no manual code edits or interactive input. Seeds, CPU execution, and numerical thread counts are fixed. Progress appears in the terminal; detailed training output is saved to `artifacts/training.log`. Missing dependencies or failed training cause a nonzero exit status with diagnostic information.

Each run overwrites the designated generated outputs. On the tested computer, warm runs take seconds; initial imports and font-cache creation can take longer. Cross-platform floating-point and neural-training results need not be bitwise identical.

## Outputs

| File | Contents |
|---|---|
| `results/tables/policy_comparison.csv` | Mean policy regret, sample standard deviation, mean value, and optimal-action accuracy for three methods |
| `results/figures/policy_regret.png` | Regret for each seed and the mean for each method |
| `results/report/summary.md` | Experimental scope and interpretation |
| `data/processed/policy_metrics.csv` | Nine seed-by-method observations |
| `data/processed/*.npz` | Generated training, validation, and test data for all three repetitions |
| `artifacts/simulation/` | Intermediate diagnostics and the full runtime configuration |

Regret is the oracle policy value minus the estimated policy's value on synthetic test outcomes. Lower regret is better. Reported standard deviations describe variation across seeds; they are not confidence intervals.

The small final table, figure, and report are committed for inspection. Generated input data and intermediate diagnostics remain ignored by Git. No precomputed result is needed as an input.

## Compare with the included reference

The repository includes a small [reference table](results/reference/policy_comparison.csv), [final figure](results/figures/policy_regret.png), and [run metadata](results/reference/metadata.json). These use the same three-seed experiment as the pipeline.

```bash
make compare
```

This command reruns the full analysis and checks the regenerated table against the fixed reference in `results/reference/`. `make reproduce` overwrites the normal final outputs but leaves the reference unchanged. Counts and method identities must match exactly; numeric summaries use `rtol=1e-5, atol=1e-7`. Cross-platform differences may require investigation. Figure bytes are not compared. See [reference documentation](results/reference/README.md).

## Tests

```bash
make test
```

Seven tests cover known-answer policy values and tied optimal actions; synthetic-data shapes, ranges and same-seed regeneration; rejection of invalid result tables; end-to-end execution with independent summary-statistic checks; integrity of the preserved source snapshot; and acceptance/rejection checks for reference comparisons. End-to-end validation writes to a new temporary directory and does not depend on existing results. See [validation](docs/validation.md) for the verification record.

## Project structure

```text
original/               preserved original-analysis snapshot and checksums
config/                 fixed experimental settings
src/                    reusable analysis code and pipeline entry point
data/raw/               documentation of the synthetic input source
data/processed/         generated data and analysis-ready metrics
artifacts/              intermediate diagnostics, configuration and logs
results/                final tables, figures, report and fixed reference
tests/                  correctness, data and pipeline validation
docs/                   provenance and validation record
```

## Original analysis and version history

The first commit preserves the selected original analysis and is tagged `original`:

`a478fdc116cbcd6fb5f261b2560909955f4ac792`

The published snapshot translates five source comments into English without changing executable Python syntax. Its manifest records both source-file and exported-file hashes. Subsequent commits record project organization, dependency pinning, automation, and validation. See [provenance](docs/provenance.md) for details. Files under `original/` are archival; use the commands in this README to run the course analysis.

## Reflection

The short [project reflection](docs/reflection.md) describes the transformation, time spent, and AI assistance.
