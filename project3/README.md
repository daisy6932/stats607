# Project 3: Faster Storey simulations

This project speeds up the Project 2 reproduction of Storey (2002), Section 5.
It reuses each dataset's p-value ordering across BH levels and evaluates
replicates in batches. The statistical design and replicate seed mapping carry
over from Project 2. The regression tests compare every output row in the full study.

On macOS ARM64, the full workflow fell from 7.08 to 5.17 seconds
(1.37× speedup); in-memory computation fell from 3.34 to 1.41 seconds
(2.37×). The benchmark data and measurement scopes are described below.

- [Baseline measurements](docs/BASELINE.md)
- [Optimization results and trade-offs](docs/OPTIMIZATION.md)
- [Performance figure](results/comparison/performance.png)

## Run the study

From the repository root, with Python 3.11 or later:

```sh
cd project3
make all
make test
```

The first run installs dependencies in `.venv`. To select a different
interpreter, run `make all PYTHON_BIN=python3.11`. The serial workflow writes raw
results, tables and scientific figures to `tmp/study/`.

| Command | Result |
|---|---|
| `make profile` | Cumulative and self-time profiles for both engines |
| `make complexity` | Timing trials as tests per dataset increase |
| `make phases` | Separate in-memory computation from CSV/gzip writing |
| `make benchmark` | Baseline and optimized timings for the complete workflow |
| `make parallel WORKERS=2` | Complete study using two scenario workers, in `tmp/parallel-study/` |
| `make stability-check` | Warning and exception counts for the main and small-tail designs |
| `make scaling` | Timings with one, two and four workers |
| `make performance` | Run the timing experiments and build the comparison figure |
| `make baseline` | Rerun the original baseline measurements in `tmp/baseline/` |
| `make clean` | Remove generated study and performance outputs under `tmp/` |

Timing outputs go to `tmp/performance/`. `TRIALS=3` is the default.
On the measured Mac, `make performance` took 81 seconds (about 1.5 minutes)
with dependencies already installed.
The measured comparison in [results/comparison/](results/comparison/) includes
individual trials and the Python and package versions. The performance figure
can also be rebuilt from those files with
`.venv/bin/python plot_performance.py results/comparison`.

The frozen Project 2 modules under `baseline/src/` are also runtime
dependencies: the optimized engine reuses the seed construction and design
configuration helpers, and both engines use the same analysis and scientific
figure modules. Regression tests compare against the original BH implementation.
The baseline study configuration and Table 2 reference are kept alongside those
modules. Environment setup and experiment commands live in the Project 3
Makefile and requirements file.
