# Project 2 baseline

The baseline is the final Project 2 repository version, commit
`b5feef94ab3b6dbe2f0fec3283cf08ef40b5f586`. The source, configurations, and
Table 2 reference under `baseline/` are copied from that revision. I used the
main study configuration: 20,000 generated datasets, 1,000 tests per dataset,
two rejection thresholds, and three methods. This yields 120,000 output rows.

## Wall time

I timed simulation, analysis, and figure generation in separate subprocesses,
as in the Project 2 Makefile. Each measured run used a new output directory
and forced simulation. The times include imports, gzip output, summaries and
plots. Dependency installation and a short warm-up run are outside the timing.
The machine ran macOS ARM64 with 10 logical CPUs and Python 3.11.18. Package
versions are in [environment.json](../results/baseline/environment.json).

| Stage | Median seconds (3 runs) |
|---|---:|
| Simulation and raw output | 5.432 |
| Analysis | 0.794 |
| Scientific figures | 0.840 |
| Complete workflow | 7.066 |

Total times were 7.065, 7.066 and 7.092 seconds. The recorded trials are in
[benchmark.json](../results/baseline/benchmark.json).

## Profile

[cProfile output](../results/baseline/profile_baseline.txt) covers one full
in-memory simulation, including construction of the result DataFrame. It
took 4.402 seconds with profiling enabled. Cumulative times can overlap.

| Operation | Calls | Cumulative seconds |
|---|---:|---:|
| BH step-up | 80,000 | 2.387 |
| NumPy argsort (including BH sorts) | 80,003 | 1.215 |
| Generate normal tests | 20,000 | 0.845 |
| Normal survival function | 20,000 | 0.679 |
| Validate p-values | 120,000 | 0.598 |
| Storey fixed region | 40,000 | 0.391 |

The same p-values are sorted four times per dataset: once per BH method
at each of two thresholds. The fixed-level BH result is also identical at
both thresholds. Normal tail probabilities are evaluated through a separate
SciPy call for each dataset. These are the first places to change.

## Complexity

Let `m` be tests per dataset, `G` the number of thresholds, and `R` the
total number of datasets. Generating the statistics, validating p-values,
and counting rejections are linear in `m`. The two BH sorts per threshold
give typical time `O(R G m log m)`; scans add `O(R G m)`. The
replicate-level result table contains `3 R G` rows.

I varied `m` from 250 to 8,000 using 100 replicates per design group,
null proportions 0.2 and 0.8, and both label modes. The log-log slope
of the median of three timings was 0.79 over this range. This is a
finite-range measurement: seed construction, Python calls and output
assembly add work that does not grow with `m`. It is not an estimate
of the asymptotic sort exponent. See [raw timings](../results/baseline/complexity.csv).

## Numerical observations

The full study and a stress case (2,000 tests, thresholds of 1e-12 and
1e-8, normal mean shift 12) produced zero numerical warnings and zero
exceptions in simulation. The all-null conditions yielded 6,000
undefined power rows in the full study and 180 in the stress case:
power has no denominator when there are no alternatives. These are
expected missing values. Each full run logged 14 Pyparsing deprecation
warnings while importing Matplotlib for figures; those are library
API warnings, not numerical failures. The simulation and analysis
subprocesses emitted no warnings. [Stability log](../results/baseline/stability.json).

From `project3/`, the original measurements can be repeated with
`python performance.py baseline --output tmp/baseline`.
