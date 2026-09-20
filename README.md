# ITR Phase 1: a reproducible NSCF simulation

Compare individualized treatment policies using the original neural supervised clustering implementation (NSCF), an initial neural Q model, and ridge regression. The experiment generates ten-treatment, two-group nonlinear synthetic data and evaluates policy regret on held-out synthetic outcomes.

## Scope and data access

This is a small reproducibility demonstration (seeds 123–125, train/validation/test sizes 160/80/1000, 15 training epochs). It is not a replication of the full paper experiments, the 50-seed benchmark, or PDX analysis. Its short optimization budget may underfit the neural models; the resulting ranking does not establish comparative performance. No patient or restricted data are included.

All input data are generated from the committed `config/experiment.json` and source code. There are no data downloads, credentials, HPC accounts, or external local-file dependencies. `make reproduce` rebuilds from the earliest stage: synthetic data generation. A separate private-data/checkpoint command is therefore unnecessary.

## Fresh setup and reproduction

System prerequisites: Python **3.11**, Git, and Make on macOS or Linux (Windows: WSL). An internet connection is needed only to install packages. No R, GPU, Slurm, notebook server, or compiler is needed when compatible wheels are available. macOS arm64 was tested; other operating systems were not tested.

After cloning this repository and entering its root directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
make reproduce
make test
```

`requirements.txt` lists the direct dependencies; `requirements-lock.txt` also pins the tested transitive dependencies. The environment itself is excluded from Git. For a platform with incompatible pinned wheels, resolve from the direct dependency file in a separate environment and record the resulting differences.

The single reproduction command is **`make reproduce`**. With the environment activated, `python src/pipeline.py` is an equivalent entry point. Without activation, use `make reproduce PYTHON=.venv/bin/python`. The pipeline logs progress and overwrites its designated generated outputs on each run. No manual edits are required. Installation time is excluded from the runtime observations in the final project documentation.

## Outputs

- `results/tables/policy_comparison.csv`: 3 methods, 3 repetitions each; means and sample standard deviations.
- `results/figures/policy_regret.png`: seed-level regret and method means.
- `results/report/summary.md`: scope and interpretation.
- `data/processed/policy_metrics.csv` and nine `.npz` files: generated metrics and train/validation/test data.
- `artifacts/simulation/`: fitted models, original implementation diagnostics, and full run configuration.
- `artifacts/training.log`: detailed progress and any original plotting warnings.

Generated outputs are present in this prepared working directory but ignored by Git. A clone recreates them with `make reproduce`; they are not precomputed dependencies. Numerical results should agree to floating-point tolerance across platforms; PNG bytes and exact neural training results are not guaranteed identical across all BLAS/hardware implementations.

## Repository map

```text
original/               untouched selected source snapshot + SHA-256 manifest
config/                 experimental settings
src/                    original reusable analysis plus new pipeline entry point
data/raw/               input-source documentation (no external raw data)
data/processed/         generated inputs and analysis-ready data
artifacts/              intermediate computations, models and logs
results/                final tables, figures and report
tests/                  meaningful correctness, data and end-to-end tests
docs/                   provenance, verification, reflection and submission notes
```

## Validation

`make test` runs four tests, including an end-to-end execution into a new temporary directory. Tests do not rely on the working directory's existing results. Validation results are recorded in the final project documentation. Missing dependencies raise an import error naming the package; install the lock file into the active environment. Training failures include a log tail and return a nonzero status.

## Original state and transformation

The original source directory had no Git repository. The first commit preserves the **selected analysis**, not a copy of the entire research workspace. Original commit: `a478fdc116cbcd6fb5f261b2560909955f4ac792` (tag `original`). `original/MANIFEST.json` records source locations and exact content checksums. Absolute source locations are provenance only and are never read by the reproduction command.

Later focused commits separate the project structure, add automation and tests, and document verification. Use `git log --oneline` to inspect them. See `docs/provenance.md` for changes from the snapshot. The other functions in the preserved research module are outside the designated course pipeline and its dependency/test contract.
