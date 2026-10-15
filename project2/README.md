# Project 2 — Simulation Study

Status: planning scaffold. The topic has not been selected or approved; simulation code and results are not yet implemented.

## Assignment checklist

- Choose Option A (reproduce a published simulation) or Option B (original simulation).
- For Option A, document the original design and results before implementation; a paper outside the supplied list requires approval.
- For Option B, consult the instructor about scope.
- Complete `ADEMP.md`, including the full simulation design matrix.
- Implement modular DGPs, methods, metrics, and orchestration in `src/`.
- Centralize parameters; document random seeds; log progress; save raw results before summarization.
- Add Makefile targets: `all`, `simulate`, `analyze`, `figures`, `clean`, `test`.
- Produce at least two figures: one diagnostic and one publication-quality; include uncertainty where appropriate.
- Implement at least three meaningful tests (DGP properties, method correctness, output validity, or reproducibility).
- Complete `ANALYSIS.md` (1–2 pages; submission list describes approximately one page).
- Final README must include setup, `make all`, estimated runtime, and 1–2 sentences of findings.
- Submit a publicly accessible repository URL, ADEMP, and a separate upload of ANALYSIS.

The source assignment filename states a deadline of October 21 at 11 pm. Confirm the course submission timezone and any updates on the course site.

## Suggested workflow

1. Confirm topic and instructor scope approval where required.
2. Finalize ADEMP and enumerate every condition before coding.
3. Run one sanity-check condition through simulation, summary, and a plot.
4. Add tests and Makefile automation, then expand to the full design.
5. Quantify Monte Carlo uncertainty, inspect diagnostics, and write the analysis from actual results.
6. Validate `make all` and `make test` from a clean environment; finalize runtime and findings.

## Candidate topic (not selected)

An original comparison of OLS slope inference using conventional standard errors versus HC3 robust standard errors under homoskedastic and heteroskedastic errors is a manageable candidate. Both methods can use the same generated datasets. Vary sample size and heteroskedasticity; measure bias, RMSE, coverage, and Type I error, with Monte Carlo uncertainty. Consult the instructor before adopting this Option B topic.

## Planned layout

```text
project2/
├── data/simulated/
├── src/                 # dgps.py, methods.py, metrics.py, simulation.py to be implemented
├── results/figures/
├── results/raw/
├── tests/
├── ADEMP.md
├── ANALYSIS.md
└── README.md
```

Setup instructions, requirements.txt, and the executable Makefile will be added when the study is implemented. There is currently no runnable analysis.
