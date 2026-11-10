# Critical analysis: Storey (2002), Section 5

## Reproduction and discrepancies

This study reproduces the main Table 2 pattern using 1000 replicates per condition. Fixed-region power is approximately 0.372 for gamma=0.01 and 0.138 for gamma=0.001, consistent with the normal-distribution calculation. BH power generally increases with the null proportion. For example, at pi0=0.1 and gamma=0.01, power is 0.3718 (MCSE 0.0005) for Storey's region and 0.0744 (MCSE 0.0006) for the paper's BH comparator. Original rounded values and all reproduction differences are retained in `results/table2_comparison.csv`.

The largest surprise was the BH result at pi0=0.8 and gamma=0.001. Table 2 reports BH power of 0.129; our main run gives 0.1127 (MCSE 0.0009). A follow-up with seed 6072027 and 10,000 replicates per design gave 0.11347 with fixed counts and 0.11334 with random labels (MCSE 0.00027 and 0.00028). The gap persisted under both designs. The paper gives no Monte Carlo error for that entry, and the source of the difference remains unclear.

## Neutrality and what the figures reveal

Storey feeds the estimated FDR for a fixed rejection region into BH on the same dataset. At pi0=0.1 and gamma=0.01, the fixed region yields empirical FDR 0.00299, compared with 0.00040 for BH. The power gain comes with more false discoveries per rejection on average. This comparison therefore cannot establish which method has higher power at the same actual FDR.

The design is favorable to tail-based null-proportion estimation: independent tests, uniform null p-values, and a common, fairly strong positive normal alternative. It includes the important BH predecessor but does not benchmark a broad range of contemporary adaptive methods. Dependence, weak or heterogeneous signals, two-sided tests, and invalid null p-values are absent. Those omissions matter in genomics, where correlated measurements and varied effect sizes are common.

The new power plot reproduces the paper's Figure 1 concept while overlaying original points and Monte Carlo intervals. The bias diagnostic exposes residual contamination of large p-values by alternatives. Before clipping, E(pi0_hat)=pi0+2*(1-pi0)*Phi(-2), so the bias near pi0=0.1 is about 0.041, despite close agreement with the original table. The empirical-FDR plot makes unequal error rates visible, which a power-only display cannot show.

## Confidence, implementation and next steps

The part of the implementation I found hardest to get right was the level
passed to BH, because a wrong choice does not fail. Section 5 gives BH the
FDR estimate from the same replicate. The pFDR estimate divides that number
by 1 − (1 − gamma)^m. At gamma=0.01 this factor is essentially one, so the
two estimates agree and half of Table 2 looks correct either way. At
gamma=0.001 the factor is 0.632 and the pFDR estimate is about 58% larger.
I settled the question with the estimated-FDR column that Table 2 prints: at
pi0=0.8 and gamma=0.001 the paper reports 0.030 and my FDR estimate averages
0.0297, while my pFDR estimate averages 0.047. A unit test with four
p-values, computed by hand, now keeps the two formulas apart. What I took
from this is to check every printed column, not only the headline one, and
to test at a setting where the candidate interpretations give different
answers.

I am confident in the implementation under the stated design. Fixed-region
power matches its closed-form value, the FDR benchmark matches an exact
enumeration on a small example, and every BH power entry except one is
within 0.003 of the printed value. For that one entry I was unsure whether
the paper fixed the number of nulls or drew labels independently, as the
errata describe, so I ran both; the follow-up above gives about 0.113 either
way, which rules out both simulation noise and the labelling choice. The
printed 0.129 also sits oddly between its neighbours, 0.101 and 0.133, so a
misprint is possible, but without the original code I cannot confirm it.
Two things could still undermine my confidence: I wrote both the code and
its tests, so a shared misreading of the paper would pass them, and the
results say nothing about correlated tests or weaker signals.
