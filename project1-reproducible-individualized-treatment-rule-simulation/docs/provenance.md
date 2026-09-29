# Analysis provenance

The source is the selected synthetic-analysis implementation from ITR Phase 1, NSCF code supplement. The course pipeline uses the original neural clustering, neural Q, ridge, synthetic-data generation, and policy evaluation code. It excludes real-data studies and larger experimental grids.

The original-analysis baseline is tagged `original`, commit `a478fdc116cbcd6fb5f261b2560909955f4ac792`. Five Chinese comments were translated into English for this export, in both the baseline and current source. Abstract syntax trees were compared before and after translation to confirm that executable Python syntax was unchanged. `original/MANIFEST.json` records each source file's original SHA-256 as `source_sha256` and its exported snapshot SHA-256 as `sha256`. The original research files remain unchanged outside this repository.

The principal analysis modification saves generated input splits as compressed NumPy files in `data/processed/`. A new pipeline entry point provides explicit seeds and CPU settings, checks result completeness and valid policy metrics, and generates the final table, figure, and report. The fixed demonstration selects three of the available methods and three repetitions. Its limited training budget is documented in the README.

The exported history retains separate stages for preserving the baseline, structuring the project, adding automation and tests, resolving dependencies, and documenting validation. Commit identifiers refer to this exported repository.
