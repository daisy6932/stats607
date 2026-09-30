# Analysis provenance

The source is the selected synthetic-analysis implementation from ITR Phase 1, NSCF code supplement. The course pipeline uses the original neural clustering, neural Q, ridge, synthetic-data generation, and policy evaluation code. It excludes real-data studies and larger experimental grids.

The original-analysis baseline is tagged `original`, commit `a478fdc116cbcd6fb5f261b2560909955f4ac792`. Five Chinese comments were translated into English for this export, in both the baseline and current source. Abstract syntax trees were compared before and after translation to confirm that executable Python syntax was unchanged. `docs/original-manifest.json` records each source file's original SHA-256 as `source_sha256` and its exported snapshot SHA-256 as `sha256`. The original research files remain unchanged outside this repository.

The principal analysis modification saves generated input splits as compressed NumPy files in `data/processed/`. A new pipeline entry point provides explicit seeds and CPU settings, checks result completeness and valid policy metrics, and generates the final table, figure, and report. The fixed demonstration selects three of the available methods and three repetitions. Its limited training budget is documented in the README.

The exported history retains separate stages for preserving the baseline, structuring the project, adding automation and tests, resolving dependencies, and documenting validation. Commit identifiers refer to this exported repository.

## Archive layout adjustment

After the original baseline was preserved, the two package directories were moved from `original/src/` directly into `original/`. The supplement README and requirements file were removed from the current archive view; both remain available in the original commit. The checksum manifest now lives in `docs/original-manifest.json`, retaining baseline paths and hashes of the removed documents. This is a later presentation change, not the historical directory structure or evidence that the source began as unstructured scripts. Python source bytes and the original tag are unchanged.

The retained original smoke script still uses its historical `src` path. To run it in the adjusted archive layout, use `PYTHONPATH="$PWD/original" bash original/scripts/run_smoke_test.sh` from the project directory with dependencies installed. Use `make reproduce` for the graded course workflow.
