# ADEMP: reproducing Storey's independent normal simulation

The study reproduces Section 5, Table 2 and Figure 1.

## Sources and interpretation

Storey, J. D. (2002). *A direct approach to false discovery rates*. JRSS B 64(3), 479–498. [Paper](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf), Sections 3–5, pp. 483–487; [author's errata](https://genomics.princeton.edu/storeylab/papers/directfdr_corrections.pdf), dated 2003-04-16. The published values are in `results/reference/table2.csv`.

Section 5 sets the fraction of nulls to pi0; the errata specify iid Bernoulli hypothesis labels for the theoretical model. The main analysis fixes the number of nulls. The sensitivity analysis draws null labels independently, following the errata. Section 5 does not specify the seed or how the BH cutoff was recorded.

## A — Aims

1. Reproduce the power patterns and numerical summaries in Table 2 for independent one-sided normal tests.
2. Measure bias and variability of the estimated null proportion and FDR, and quantify Monte Carlo uncertainty.
3. Evaluate whether the paper's same-data, estimated-level BH comparison supports a general claim of superiority. Report empirical FDR for BH as well as Storey's fixed region, and include conventional BH at a prespecified 0.05 as a contextual baseline.
4. Check the effect of random rather than fixed null counts, plus all-null and all-alternative sanity checks.

## D — Data-generating mechanisms

Each replicate contains m=1000 independent statistics. Null statistics are N(0,1); alternatives are N(2,1). Compute one-sided p=normal survival function(Z). The primary design has m0=round(m*pi0) nulls and m1=m-m0 alternatives, in a fixed order (all methods are permutation invariant). The sensitivity design independently labels each test null with probability pi0. No estimated parameters from the generated sample enter p-value computation.

Use 1000 independent replicates per condition, as in Section 5, lambda=0.5, and gamma in {0.01,0.001}. The two gamma analyses share the same dataset within each pi0/mode/replicate to improve paired comparisons. Draws for distinct pi0/mode groups are independent. Seed 6072026; NumPy PCG64 via SeedSequence([seed, mode_code, round(1000*pi0), replicate]), where fixed=0 and iid=1. Each replicate is independently regenerable; reordering conditions or extending replication counts does not alter existing draws.

### Complete design matrix

All rows have m=1000, alternative mean=2, lambda=0.5, 1000 replicates. Each gamma cell below is a separate condition, so there are 40 conditions: 18 primary, 18 sensitivity, 4 boundary. Fixed/BH-at-estimated-level/BH-at-0.05 use the same data in each condition.

| pi0 | Fixed-count primary gamma | IID-label sensitivity gamma | Fixed-count boundary gamma |
|---|---|---|---|
| 0.0 | — | — | 0.01, 0.001 |
| 0.1 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.2 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.3 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.4 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.5 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.6 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.7 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.8 | 0.01, 0.001 | 0.01, 0.001 | — |
| 0.9 | 0.01, 0.001 | 0.01, 0.001 | — |
| 1.0 | — | — | 0.01, 0.001 |

The executable config is `config/study.json`; analysis exports one explicit row per condition to `results/design_matrix.csv`. A separate small `config/smoke.json` is for pipeline checks only.

## E — Estimands / targets

- pi0: population null proportion; equal to the actual proportion only in fixed-count designs.
- Power: probability of rejecting a false null. Its analytic fixed-region target is b=SF(ISF(gamma)-2).
- FDR: E[V/max(R,1)], with V false rejections and R total rejections.
- pFDR: E[V/R | R>0], kept distinct from FDR. Empirically average FDP only over replicates with rejections.
- Fixed-count exact FDR: m0*gamma times the integral from 0 to 1 of (1-gamma+gamma*t)^(m0-1)*(1-b+b*t)^m1 dt (zero if m0=0). Pr(R>0)=1-(1-gamma)^m0*(1-b)^m1; pFDR=FDR/Pr(R>0).
- For iid labels, q=pi0*gamma+(1-pi0)*b; pFDR=pi0*gamma/q and FDR=pFDR*[1-(1-q)^m]. Analytic targets validate the simulation; they are never given to either method.

## M — Methods

1. **Storey fixed region:** reject p<=gamma. W=#{p>lambda}; retain the untruncated diagnostic W/[m*(1-lambda)] and use pi0_hat=min(1,W/[m*(1-lambda)]) in the estimator, following the erratum's constraint. Estimate FDR=min(1,pi0_hat*m*gamma/max(R,1)); estimate pFDR by additionally dividing the uncapped FDR expression by 1-(1-gamma)^m and then capping at one.
2. **BH at estimated level (paper comparison):** on the same replicate set alpha to the Storey FDR estimate for that replicate. Sort p values; k=max{i:p_(i)<=alpha*i/m}; reject through k, or none if the set is empty. Define cutoff as largest rejected observed p, or zero. This convention might differ from Table 2's undocumented cutoff convention.
3. **BH at fixed 0.05 (extension):** same step-up rule with a prespecified level. The fixed level provides a comparison with the replicate-specific BH level used in the paper.

## P — Performance measures and reporting

The analysis averages replicate-level power and FDP, with pFDR calculated among replicates with at least one rejection. Monte Carlo standard errors use the sample standard deviation divided by the square root of the valid replicate count. Pointwise 95% intervals use the mean ± 1.96 MCSE. Power differences use paired replicates. Estimator bias and RMSE use the analytic targets.

The power figure overlays published Table 2 values. The diagnostics show bias in the estimated null fraction and FDR. Two additional figures show empirical FDR and sensitivity to label assignment.
