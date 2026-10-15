# Source audit before implementation

2026-10-15: read the author's PDF and one-page errata. URLs and page references are in ADEMP.md. Copyrighted PDFs are linked rather than redistributed.

- Scope: Section 5 / Table 2 / Figure 1; m=1000, 1000 repetitions, normal alternative mean 2, one-sided upper-tail tests, lambda=0.5, two gamma values and nine pi0 values.
- Record both FDR and pFDR. Section 5 uses the FDR estimator (equation 10), without the extra pFDR correction.
- BH uses a different estimated alpha in every replicate, computed from the same p-values. It is not fixed-alpha BH and is not an independent validation of error control.
- Section 5 describes a set proportion of nulls; the errata's theoretical assumptions instead randomize the labels. Preserve this distinction with fixed-count primary and iid sensitivity analyses.
- Keep raw and capped pi0 estimates. The errata explicitly constrain pi0_hat <=1 in the Section 4 comparison; boundary conditions can therefore behave differently from the uncapped theoretical bias statement.
- Table 2 is transcribed before implementation in results/reference/table2.csv. Its entries are rounded and do not include Monte Carlo standard errors. Gamma-hat's exact convention is unspecified; use largest rejected p, document it, and do not force a numerical match.
- The paper's Table 2 has BH power 0.129 at gamma=0.001, pi0=0.8. Preserve that entry as printed even if the new run differs.
- The errata qualify claims about optimal lambda selection; adaptive lambda is outside this project's scope.
