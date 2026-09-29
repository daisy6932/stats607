# Committed numerical reference

`policy_comparison.csv` is the reference from the 2026-09-29 CPU run with the committed configuration and dependency lock. `metadata.json` records its environment and configuration.

From the project folder, after environment setup:

```bash
make compare
```

This reruns the analysis from synthetic data and compares the regenerated `../tables/policy_comparison.csv` with this fixed reference. It checks method identities and repetition counts exactly and numerical values with `rtol=1e-5, atol=1e-7`. The pipeline never reads the reference while generating results and never overwrites it. A mismatch exits with an error; investigate platform, dependencies, settings or code changes rather than silently replacing the reference.

The committed final figure in `../figures/` is also available for visual comparison. PNG byte equality is not required because rendering can differ across platforms. The reference is not a data checkpoint or evidence of general model superiority.
