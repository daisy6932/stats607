# Critical analysis: Storey (2002), Section 5

## Reproduction and discrepancies

This study reproduces the main Table 2 pattern using 1000 replicates per condition. Fixed-region power is approximately 0.372 for gamma=0.01 and 0.138 for gamma=0.001, consistent with the normal-distribution calculation. BH power generally increases with the null proportion. For example, at pi0=0.1 and gamma=0.01, power is 0.3718 (MCSE 0.0005) for Storey's region and 0.0744 (MCSE 0.0006) for the paper's BH comparator. Original rounded values and all reproduction differences are retained in `results/table2_comparison.csv`.

One discrepancy is substantial: at pi0=0.8 and gamma=0.001, Table 2 prints BH power 0.129, whereas this run gives 0.1127 (MCSE 0.0009). The iid-label sensitivity gives 0.1152 (MCSE 0.0009), so changing label generation does not eliminate it. The original code and seed are unavailable, and its Monte Carlo uncertainty is unreported; the cause remains unresolved. The published entry is preserved rather than corrected or used to tune the simulation. Smaller differences can reflect rounding, independent Monte Carlo variation, or undocumented implementation choices, including the BH cutoff convention.

## Neutrality and what the figures reveal

The central comparison is not a conventional comparison at matched realized FDR. Storey fixes a rejection region, estimates its FDR, then feeds that estimate from the same dataset to BH. At pi0=0.1 and gamma=0.01, empirical FDR is 0.00299 for the fixed region and 0.00040 for BH. Thus, the observed power advantage accompanies a higher actual error rate; it does not by itself establish superiority at equal realized FDR. The added prespecified-level BH baseline puts this difference in context, but also has a different error target and is not an equal-FDR comparison.

The design is favorable to tail-based null-proportion estimation: independent tests, uniform null p-values, and a common, fairly strong positive normal alternative. It includes the important BH predecessor but does not benchmark a broad range of contemporary adaptive methods. Dependence, weak or heterogeneous signals, two-sided tests, and invalid null p-values are absent. Those omissions matter in genomics, where correlated measurements and varied effect sizes are common.

The new power plot reproduces the paper's Figure 1 concept while overlaying original points and Monte Carlo intervals. The bias diagnostic exposes residual contamination of large p-values by alternatives. Before clipping, E(pi0_hat)=pi0+2*(1-pi0)*Phi(-2), so the bias near pi0=0.1 is about 0.041, despite close agreement with the original table. The empirical-FDR plot makes unequal error rates visible, which a power-only display cannot show.

## Confidence, implementation and next steps

The main implementation challenge was distinguishing FDR from pFDR and preserving the replicate-specific BH input. Tests check these formulas, the BH step-up rule, exact binomial FDR calculations, DGP properties, output validity and reproducibility. Stable replicate streams and saved raw counts make results auditable. All 26 tests pass; the complete pipeline runs from clean outputs. Confidence is strongest in the defined independent-normal design, not in general claims about real applications.

Boundary checks are instructive: with all nulls and gamma=0.001, fixed-region FDR is about 0.635, close to its exact value 0.6323, whereas pFDR is one. Clipping estimates can change bias, so uncapped theoretical conservativeness should not be transferred automatically to the implemented capped estimator. Zero observed Monte Carlo variability at a boundary is not proof of zero underlying uncertainty.

A stronger follow-up would compare power against achieved FDR across thresholds, using separate calibration simulations before evaluation, then add correlated tests and weaker heterogeneous signals. More replicates would improve precision for rare false discoveries. Adaptive lambda and bootstrap intervals are useful extensions, but are outside this scoped reproduction; the errata also qualify claims of optimal lambda selection.
