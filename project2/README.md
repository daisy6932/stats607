# Project 2 — Reproducing Storey (2002)

A reproducible simulation study of false discovery rate estimation and power, reproducing Section 5, Table 2 and Figure 1 of [Storey (2002), *A direct approach to false discovery rates*](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf), using the [author's errata](https://genomics.princeton.edu/storeylab/papers/directfdr_corrections.pdf) to guide the label sensitivity analysis.

The primary reproduction has 18 conditions: independent normal tests, 1000 hypotheses per replicate, 1000 replicates, alternative mean 2, lambda=0.5, pi0=0.1,…,0.9, and gamma=0.01/0.001. An additional 18 conditions use random labels; four boundary conditions check all-null/all-alternative behavior. There are 20,000 independent generated datasets, shared across the two gamma thresholds, and 120,000 method-level raw rows.

**Key findings:** Storey's fixed-region power is approximately 0.372 and 0.138 for the two thresholds, reproducing the paper's main pattern; one printed BH power entry differs materially. The paper's BH comparator uses a same-data estimated level, and observed FDR differs between methods, so the power advantage does not establish superiority at matched realized FDR.

## Run the study

From the repository root:

```bash
cd project2
make setup
make all
make test
```

Setup creates `.venv`. Clean installs and full runs passed on Python 3.13.12
and 3.15.2 on macOS ARM64. To select an interpreter, run
`make setup PYTHON_BIN=python3.13`. For the original Python 3.13 package versions,
use `make setup PYTHON_BIN=python3.13 REQUIREMENTS=requirements-lock.txt VENV=.venv-locked`
and `make all VENV=.venv-locked`.

A clean simulation, analysis, and figure run takes about 8–15 seconds on macOS ARM64, excluding dependency installation and first-time font setup.

| Command | Output |
|---|---|
| `make simulate` | Replicate data and run metadata |
| `make analyze` | Summary tables and Table 2 comparison |
| `make figures` | Four figures in PNG and SVG |
| `make smoke` | Small complete run in `tmp/smoke/` |
| `make clean` | Removes generated results and keeps the Table 2 reference |

[ADEMP.md](ADEMP.md) explains the design and [config/study.json](config/study.json)
defines the conditions. In each replicate the paper's BH comparator uses Storey's
FDR estimate as its input level. The study also includes BH at a fixed level of 0.05.

## Outputs

- [ANALYSIS.md](ANALYSIS.md): critical analysis, including discrepancies and limitations; upload separately as required.
- [results/table2_comparison.csv](results/table2_comparison.csv): all six numerical columns compared with the transcribed published values, with differences and Monte Carlo standard errors.
- [results/summary.csv](results/summary.csv): power, empirical FDR/pFDR, cutoff and rejection count summaries with valid sample sizes and pointwise 95% Monte Carlo intervals.
- [results/diagnostics.csv](results/diagnostics.csv): bias, RMSE and clipping diagnostics.
- [results/paired_power.csv](results/paired_power.csv): within-dataset power differences and Monte Carlo uncertainty.
- [results/design_matrix.csv](results/design_matrix.csv): one row per simulation condition.
- [results/raw/](results/raw/): replicate-level CSV and provenance; [schema](docs/output_schema.md).

### Publication figure: power reproduction

![Power compared against original Table 2, with Monte Carlo intervals](results/figures/power_comparison.png)

### Diagnostic figure: estimator bias

![Bias of null-proportion and FDR estimates](results/figures/estimator_diagnostics.png)

Additional figures show [empirical FDR](results/figures/empirical_fdr.png) and [label sensitivity](results/figures/label_sensitivity.png). SVG versions in the same folder provide scalable figures. Error bars show 95% Monte Carlo intervals.

## Project structure

```text
project2/
├── config/                 # full and smoke JSON designs
├── data/simulated/         # optional full-vector cache; unused (seed regeneration instead)
├── docs/                   # source audit, schema, validation record
├── src/
│   ├── dgps.py             # DGP and independent replicate streams
│   ├── methods.py          # Storey estimates and BH
│   ├── metrics.py          # realized metrics and analytic benchmarks
│   ├── simulation.py       # orchestration, caching, provenance
│   ├── analyze.py          # validation, summaries, paired comparisons
│   └── figures.py          # diagnostic and publication figures
├── results/
│   ├── raw/               # compressed replicates + metadata
│   ├── reference/         # Table 2 transcription, preserved by clean
│   └── figures/           # PNG/SVG
├── tests/
├── Makefile
├── ADEMP.md
├── ANALYSIS.md
├── requirements.txt
├── requirements-lock.txt
└── .gitignore
```
