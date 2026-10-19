# Validation record — 2026-10-15

Environment: Python 3.13.12, macOS ARM64; dependency versions are pinned in requirements files and recorded in `results/raw/metadata.json`.

- `make test PYTHON=.venv/bin/python`: **26 passed** (final run 0.74 seconds).
- `make smoke PYTHON=.venv/bin/python`: full small pipeline, 480 raw rows, four PNG/SVG figures.
- `make clean PYTHON=.venv/bin/python` followed by `make all PYTHON=.venv/bin/python`: succeeded, producing 120,000 raw rows across 40 conditions and three methods, summaries, and four PNG/SVG figures. Full wall time 7.23 seconds; simulation/raw writing 4.83 seconds. Installation and initial font-cache building excluded.
- Repeated `make all`: verified raw cache reused; summaries and figures regenerated successfully.
- `python -m pip check`: no broken requirements.
- Raw SHA-256, all recorded source fingerprints, row count, and local documentation links verified.
- `git diff --check`: passed.
- All four figures visually inspected for labels, comparison points, uncertainty intervals, legends and captions.

Tests cover BH step-up behavior and boundaries; Storey FDR/pFDR arithmetic; clipping and zero rejections; invalid p-values; DGP distribution and labels; analytic FDR against exact enumeration; all-null identities; Monte Carlo summaries; exact same-seed reproduction; invariance to threshold order and extension of replication count; raw validity and missing outputs; end-to-end summaries; analytic versus simulated performance; cache reuse and corruption recovery.

Rounded published values are not treated as exact test targets. The discrepancy at pi0=0.8, gamma=0.001 is documented in ANALYSIS.md. Repository visibility remains private; the public-access submission requirement is unresolved pending the user's decision about the existing course archive.

## Requirement audit — 2026-10-19

Re-read the full local Unit 2 assignment and inspected the committed implementation and documents. Audited commit: `f15badc506787bf2347c93aa962219bb74d5683d`.

| Assignment requirement | Evidence | Status |
|---|---|---|
| Option A paper and pre-implementation design/results | Storey is on the instructor's list; design/source notes and reference Table 2 were committed in `25a4d1c`, before implementation commit `31f0dcb` | Complete |
| ADEMP, reproducible DGP description, all condition combinations | ADEMP.md contains all five components, formulas, replication/seed protocol and design table; results/design_matrix.csv expands the 40 conditions | Complete |
| Required project structure | data/simulated, src/dgps.py, methods.py, metrics.py, simulation.py, results/raw, results/figures, tests, Makefile, documents, requirements and .gitignore are tracked | Complete |
| Modular functions, seed management, config, progress, intermediate raw output | Inspected src modules and JSON config; logging reports each group; raw CSV is saved before analysis; provenance and cache integrity are recorded | Complete |
| All six required Makefile targets | all, simulate, analyze, figures, clean and test are present; clean/all/test executed on an exported committed snapshot | Complete |
| At least two interpretable figures, including diagnostic/publication plots | Four figures in PNG/SVG; primary power plot and bias diagnostic have labels, facets, legends and Monte Carlo intervals | Complete |
| At least three meaningful tests | 26 pytest cases passed again, including DGP, methods, analytic benchmarks, output validity and reproducibility | Complete |
| ANALYSIS.md, approximately 1–2 pages | About 582 words; covers reproduction, discrepancy, neutrality, realism, figures, implementation, confidence and future design changes | Complete as a file; separate course upload still required |
| README description, setup, make all, runtime and short findings | All present; direct and transitive dependencies pinned | Complete |
| Public repository URL | GitHub API confirms daisy6932/stats607 is PRIVATE as of this audit | Not complete; visibility change needs the user's decision |
| Separate ANALYSIS.md upload and course submission | File is ready; no course-site submission was performed | Pending |

Fresh-output verification: exported the committed project with git archive into a temporary directory (no working-tree/untracked files), ran `make clean`, `make all` and `make test`, using the existing pinned Python environment. All commands succeeded. There were 120,000 raw rows and 26 passing tests (0.77 seconds); complete audit execution including first-time font-cache setup and tests took 17.38 seconds. This was a clean source/output check, not a new dependency installation or a different-platform test.

Every regenerated top-level result CSV and the compressed raw replicate file matched the committed version byte-for-byte. Raw SHA-256: `d4a858ccfd5f5c36426af3e83003374432fa7354e9b4896ff1fef17e6772f835`. `pip check` also reports no broken requirements. The source and remote commit matched before this documentation-only audit update.

The unresolved published BH entry is a documented reproduction difference, not a missing assignment component. Full reproduction of all sections of Storey's paper is not claimed; the project explicitly scopes its reproduction to Section 5.

## Public-access update — 2026-10-19

At the user's explicit request, the repository was changed to PUBLIC. The GitHub API confirmed the new visibility. This resolves the public-repository requirement noted in the audits above. Course-site submission and separate ANALYSIS.md upload remain for the student.
