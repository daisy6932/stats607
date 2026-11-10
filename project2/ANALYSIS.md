# Critical analysis: Storey (2002), Section 5

## Reproduction and discrepancies

This study reproduces the main Table 2 pattern using 1000 replicates per condition. Fixed-region power is approximately 0.372 for gamma=0.01 and 0.138 for gamma=0.001, consistent with the normal-distribution calculation. BH power generally increases with the null proportion. For example, at pi0=0.1 and gamma=0.01, power is 0.3718 (MCSE 0.0005) for Storey's region and 0.0744 (MCSE 0.0006) for the paper's BH comparator. Original rounded values and all reproduction differences are retained in `results/table2_comparison.csv`.

At pi0=0.8 and gamma=0.001, Table 2 reports BH power of 0.129; our main run gives 0.1127 (MCSE 0.0009). A follow-up with seed 6072027 and 10,000 replicates per design gave 0.11347 with fixed counts and 0.11334 with random labels (MCSE 0.00027 and 0.00028). The gap persisted under both designs. The paper gives no Monte Carlo error for that entry, and the source of the difference remains unclear.

## Neutrality and what the figures reveal

Storey feeds the estimated FDR for a fixed rejection region into BH on the same dataset. At pi0=0.1 and gamma=0.01, the fixed region yields empirical FDR 0.00299, compared with 0.00040 for BH. The power gain comes with more false discoveries per rejection on average. This comparison therefore cannot establish which method has higher power at the same actual FDR.

The design is favorable to tail-based null-proportion estimation: independent tests, uniform null p-values, and a common, fairly strong positive normal alternative. It includes the important BH predecessor but does not benchmark a broad range of contemporary adaptive methods. Dependence, weak or heterogeneous signals, two-sided tests, and invalid null p-values are absent. Those omissions matter in genomics, where correlated measurements and varied effect sizes are common.

The new power plot reproduces the paper's Figure 1 concept while overlaying original points and Monte Carlo intervals. The bias diagnostic exposes residual contamination of large p-values by alternatives. Before clipping, E(pi0_hat)=pi0+2*(1-pi0)*Phi(-2), so the bias near pi0=0.1 is about 0.041, despite close agreement with the original table. The empirical-FDR plot makes unequal error rates visible, which a power-only display cannot show.

## Confidence, implementation and next steps

The key implementation choice was which estimate to pass to BH. Section 5 uses the estimated FDR from that replicate. The pFDR estimate has an additional correction for the chance of making any rejection; using it would change the BH input. A four-p-value test checks both formulas by hand. Separately, enumerating the possible rejection counts in a small example checks the FDR benchmark used to judge the simulations.

Boundary checks are instructive: with all nulls and gamma=0.001, fixed-region FDR is about 0.635, close to its exact value 0.6323, whereas pFDR is one. Clipping estimates can change bias, so uncapped theoretical conservativeness should not be transferred automatically to the implemented capped estimator.

A stronger follow-up would compare power against achieved FDR across thresholds, using separate calibration simulations before evaluation, then add correlated tests and weaker heterogeneous signals. More replicates would improve precision for rare false discoveries.
