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
