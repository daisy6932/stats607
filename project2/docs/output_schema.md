# Outputs and regeneration

`results/raw/replicates.csv.gz` has one row per condition, replicate and method. There are 40 conditions x 1000 replicates x 3 methods. Two gamma conditions share each generated dataset. The prespecified BH baseline is consequently repeated across gamma values by design; these rows must not be treated as independent replications.

| Fields | Meaning |
|---|---|
| mode, role | fixed/iid labels; primary/sensitivity/boundary |
| pi0, gamma, m, effect, lambda | Generating probability or fixed proportion, p-value threshold, number of tests, normal alternative mean, tuning parameter |
| seed, replicate | Master seed and zero-based replicate; stream key additionally uses mode and pi0 as defined in ADEMP |
| method | storey_fixed, bh_estimated, bh_fixed |
| cutoff | gamma for fixed region; largest rejected observed p for BH; zero if BH rejects none |
| alpha | BH input level; blank for fixed region (no prespecified FDR level) |
| pi0_raw, pi0_hat | Untruncated and capped Storey null-proportion estimates |
| fdr_hat, pfdr_hat | Storey estimates for that dataset/threshold, attached to all method rows for pairing; not estimates made by BH |
| rejections, false_rejections, true_rejections | R, V, R-V |
| alternatives | Actual alternative count, random in iid mode |
| fdp | V/max(R,1) |
| power | True rejections / alternatives; blank if no alternatives |

`metadata.json` records config content/path/hash, source SHA-256 hashes, versions, simulation time, row count and raw-file hash. NumPy's PCG64 stream is selected explicitly. Whole p-value vectors are not saved; regenerate them from the recorded seed keys and DGP. `data/simulated/` is reserved for an optional full-vector cache.

In the summary, `metric=fdp` is the mean FDP across replicates (empirical FDR).
For `metric=pfdr`, the mean uses replicates with at least one rejection. Summary
rows report the mean, MCSE, valid count and pointwise 95% Monte Carlo interval.
In the diagnostics, `mean` is the bias and `estimate_mean` is the mean estimate.
