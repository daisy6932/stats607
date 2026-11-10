# Critical analysis: Storey (2002), Section 5

## Reproduction and discrepancies

This study reproduces the main Table 2 pattern using 1000 replicates per condition. Fixed-region power is approximately 0.372 for gamma=0.01 and 0.138 for gamma=0.001, consistent with the normal-distribution calculation. BH power generally increases with the null proportion. For example, at pi0=0.1 and gamma=0.01, power is 0.3718 (MCSE 0.0005) for Storey's region and 0.0744 (MCSE 0.0006) for the paper's BH comparator. Original rounded values and all reproduction differences are retained in `results/table2_comparison.csv`.

The largest surprise was the BH result at pi0=0.8 and gamma=0.001. Table 2 reports BH power of 0.129; our main run gives 0.1127 (MCSE 0.0009). A follow-up with seed 6072027 and 10,000 replicates per design gave 0.11347 with fixed counts and 0.11334 with random labels (MCSE 0.00027 and 0.00028). The gap persisted under both designs. The paper gives no Monte Carlo error for that entry, and the source of the difference remains unclear.

## Neutrality and what the figures reveal

Storey feeds the estimated FDR for a fixed rejection region into BH on the same dataset. At pi0=0.1 and gamma=0.01, the fixed region yields empirical FDR 0.00299, compared with 0.00040 for BH. The power gain comes with more false discoveries per rejection on average. This comparison therefore cannot establish which method has higher power at the same actual FDR.

The design is favorable to tail-based null-proportion estimation: independent tests, uniform null p-values, and a common, fairly strong positive normal alternative. It includes the important BH predecessor but does not benchmark a broad range of contemporary adaptive methods. Dependence, weak or heterogeneous signals, two-sided tests, and invalid null p-values are absent. Those omissions matter in genomics, where correlated measurements and varied effect sizes are common.

The new power plot reproduces the paper's Figure 1 concept while overlaying original points and Monte Carlo intervals. The bias diagnostic exposes residual contamination of large p-values by alternatives. Before clipping, E(pi0_hat)=pi0+2*(1-pi0)*Phi(-2), so the bias near pi0=0.1 is about 0.041, despite close agreement with the original table. The empirical-FDR plot makes unequal error rates visible, which a power-only display cannot show.

## Confidence, implementation and next steps

The hardest part to verify was what BH receives in each replicate. Both FDR and pFDR estimates are valid inputs to the software, so a successful run would not expose a mix-up. Section 5 specifies the FDR estimate, and the four-p-value unit test separates the formulas: it gives 0.010 for FDR and about 0.254 for pFDR. I take that as a reason to check the quantity passed between methods as carefully as the final output.

I am confident in the main pattern for independent normal tests. The normal-tail calculation gives fixed-region power of about 0.372 and 0.138, and the simulated values agree. Most Table 2 values are close, and a small-case enumeration independently checks the FDR benchmark. At gamma=0.001, the published BH values at pi0=0.7 and 0.9 are 0.101 and 0.133. The follow-up result of about 0.1134 fits the progression between those rows more closely than the printed 0.129 does. This makes a reporting error plausible, but the table alone cannot establish one. A hand calculation can verify the formulas used here while still sharing the same interpretation of the paper. The original code and Monte Carlo uncertainty would help distinguish a reporting error from a difference in implementation.

A stronger follow-up would compare power against achieved FDR across thresholds, using separate calibration simulations before evaluation, then add correlated tests and weaker heterogeneous signals. More replicates would improve precision for rare false discoveries.
