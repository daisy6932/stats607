# Optimizing the Storey simulation

The complete workflow went from **7.08 to 5.17 seconds** (1.37× speedup,
27% less time). The in-memory calculation improved by 2.37×, from
3.34 to 1.41 seconds, in a separate unprofiled timing experiment.
CSV formatting, gzip compression and writing still took about 1.6 seconds
per run. Those costs, plus imports and the unchanged analysis and figures,
explain why the whole workflow improved less than the computation.

Both timing experiments used medians of three runs per version on the same
macOS ARM64 machine and Python environment. Full-workflow trials used separate
subprocesses and fresh output directories. The phase experiment measured the
in-memory call and then timed writing that result to a compressed CSV; its
computation timing excludes imports and file output.
[Whole-workflow timings](../results/comparison/benchmark.json) ·
[Compute and write timings](../results/comparison/phases.json) ·
[Performance figure](../results/comparison/performance.png)

Repeat the phase measurement with `make phases`; it writes
`tmp/performance/phases.json`.

## Reuse the ordering for BH

The baseline profile showed 80,000 BH calls. At each of two gamma values it
sorted the same dataset once for Storey's estimated level and once for fixed
alpha. The fixed-alpha result was also identical across gamma values.

Before, inside the gamma loop:

```python
bh_at_estimated_fdr, estimated_cutoff = bh(pvalues, estimates["fdr_hat"])
bh_at_fixed_level, baseline_cutoff = bh(pvalues, config["baseline_alpha"])
```

After, assemble the estimated levels and fixed alpha, then reuse the ordering:

```python
levels = np.column_stack((fdr_hat, np.full(batch_rows, config["baseline_alpha"])))
bh_total, bh_false, bh_cutoff = bh_counts(pvalues, is_null, levels)
```

In `bh_counts`, one row sort orders both the p-values and the null labels.
A prefix sum of the ordered null labels gives the false rejection count at
each selected rank. The fixed-alpha counts are broadcast over gamma when
building the result table. This takes the number of dataset sorts from
80,000 to 20,000. Across the paired profiles, cumulative NumPy argsort time
fell from 1.354 to 0.595 seconds (including any sorts outside BH).
The sort term changes from `O(R G m log m)` to `O(R m log m)`; scanning the
levels still takes `O(R G m)`. The prefix table uses extra temporary memory
but avoids rebuilding and summing rejection masks for every BH level.

The cost is readability:
`bh_counts` works with row sorting, prefix counts and rank-indexed lookups,
whereas the original `bh` returns a simple rejection mask. Handling a zero
selected rank and aligning the null prefix with that rank require particular
care. Keeping the frozen baseline also means maintaining two BH
implementations with the same semantics. The exact row comparisons and
tied-p-value tests make mismatches visible when the optimized code changes.

## Evaluate replicates in batches

The original profile attributed 0.746 seconds to 20,000 calls to the
normal survival function. Storey and BH also repeatedly built masks one
replicate at a time.

Before, generate and evaluate each dataset separately:

```python
pvalues, is_null = generate_normal_tests(rng, config["m"], pi0, config["effect"], mode)
storey_reject, estimates = storey_fixed(pvalues, gamma, config["lambda"])
```

After, retain the separate seed stream for each replicate but batch the
tail transform and the threshold counts:

```python
pvalues, is_null = generate_batch(config, mode, pi0, replicates)
fixed_mask = pvalues[:, None, :] <= gammas[None, :, None]
fixed_counts = fixed_mask.sum(axis=2)
```

`generate_batch` calls `norm.sf(statistics)` once per block. With 64
replicates per block, the profile shows 320 calls and 0.280 seconds
in that function. The broadcast arrays have replicate, cutoff and test axes,
and the output table keeps the original replicate, cutoff and method order.
The extra working arrays scale with `O(B (G+1) m)` per worker for block size
`B`. Limiting the block size keeps temporary storage from growing with the
entire study. The two optimizations interact, so I use the sort and survival
function profile entries to show their specific effects rather than summing
them as separate wall-time gains.

## Split scenarios across processes

The serial version walks the design groups in order. Each group has separate
replicate streams, so it can be sent to one worker without sharing RNG state.

```python
# Serial
groups = list(map(run_group, tasks))
# Parallel, inside a spawn-based ProcessPoolExecutor
groups = list(pool.map(run_group, tasks))
```

The full study took a median 1.43 s in memory with one worker, 1.37 s
with two, and 1.11 s with four (1.28× over optimized serial). The
[worker timings](../results/comparison/parallel.csv) include process startup
and transfer of the result tables. They exclude gzip and plots. The speedup
is modest because each task is already short and each process needs its own
imports and memory. I kept serial as the default and exposed the process
version with `make parallel WORKERS=4`.

## Scaling and correctness

The [log-log comparison](../results/comparison/performance.png) varies the
number of tests from 250 to 8,000. The optimized version was faster at all
six measured sizes. At 1,000 tests, the median in-memory times were 0.068 s
and 0.030 s; at 8,000 they were 0.556 s and 0.233 s. The slopes
in the figure are 1.13 (baseline) and 1.01 (optimized), fitted only to
m = 2,000, 4,000 and 8,000. All six measured sizes remain visible, but
the dashed fits cover only those three large sizes. Small datasets have
a larger contribution from fixed overhead, which bends the full-range
curve. A three-point fit over this finite range is a scaling diagnostic,
not a proof of asymptotic complexity. With the number of gamma values
fixed, both engines still have the same asymptotic order in m.

The initial baseline experiment in BASELINE.md fitted all six sizes and
gave 0.79. An independent baseline measurement in the paired comparison
gave 0.74 with the same full-range fitting rule; its optimized slope was
0.89. The two baseline numbers come from different timing trials, not
different source code. The updated [fit metadata](../results/comparison/complexity_fit.json)
records both the full-range slopes and the large-size fits used in the figure.

The paired cProfile runs show 4.846 versus 1.704 seconds (2.84×) with
profiling enabled. These instrumented times are separate from the
unprofiled compute and complete-workflow timings reported above.

The full-study regression matches every value in the 120,000 rows exactly.
Tests also compare summary tables, process ordering, tied p-values,
no rejections, all-null power and a small-tail design. Both engines
produced zero numerical warnings and exceptions in the
[stability runs](../results/comparison/stability.json). The optimized version
retains the baseline's `log1p`/`expm1` calculation and the same convention
for undefined power.

## Lessons learned

Removing repeated sorting and batching the array work gave the best return.
The unprofiled computation time fell by nearly two seconds without changing
a single output value. The trade-off is more complicated indexing and array
shapes; that is why the small BH edge cases matter alongside the full-study
regression.

The shift in the bottleneck was more surprising than the total speedup.
Formatting and writing the same 120,000 rows took 1.59 seconds after
optimization, slightly longer than the 1.41-second calculation. Making the
numerical part faster again would leave most of that work untouched.
Before adding more computational machinery, I would profile CSV formatting
and gzip separately and check whether a different storage format is worth
losing the convenience of a readable CSV.

Four workers saved about 0.32 seconds over optimized serial computation,
despite adding process startup, data transfer and separate copies of the
working arrays. Two workers saved only about 0.06 seconds. For this study,
that is a small return for the extra deployment and debugging burden, so
serial remains the default. Parallel execution would be worth reconsidering
if scenarios were larger or more expensive. It does not replace the
simpler gains from eliminating repeated work.
