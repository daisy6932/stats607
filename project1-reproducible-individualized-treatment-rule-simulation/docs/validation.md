# Validation record

Platform: macOS arm64, Python 3.11. The specified environment was created from `requirements-lock.txt` on 2026-09-27; `pip check` reported no broken requirements.

The pipeline was run from synthetic inputs, and its final CSV agreed with the previously verified implementation within relative and absolute tolerance `1e-10`. The final figure was visually checked for readable labels and unclipped content. A separate clean clone also reproduced the outputs without copied datasets or generated results.

Five tests validate:

1. Hand-computable policy values and regret, including tied optimal actions.
2. Synthetic data shapes, finite values, action ranges, optimal outcomes, and exact same-seed generation.
3. Rejection of duplicate observations, inconsistent regret and nonfinite metrics.
4. Full pipeline execution in an empty temporary output directory, with independent checks of means and sample standard deviations.
5. SHA-256 integrity of all 27 files in the original-analysis snapshot.

The Codex filesystem sandbox restricts OpenMP shared memory and can interrupt the nested training process during testing. Full tests are run outside that sandbox. Initial font-cache and auxiliary diagnostic-plot layout warnings do not prevent generation of the designated final figure. Cross-platform execution and paper-scale statistical claims have not been validated.

The English-language publication export was revalidated before upload: all five tests passed, and the regenerated summary table matched the earlier tested project within `1e-10`.

## Reference comparison update (2026-09-29)

The small final CSV, PNG, and report are now committed, together with a separate reference table and its configuration/environment metadata. `make compare` regenerated the analysis and passed the numerical comparison (`rtol=1e-5`, `atol=1e-7`). The full suite now has **7 passing tests**, including acceptance of reordered reference rows and rejection of a deliberately changed regret value. Pytest completed in 5.14 seconds on the tested host. The reference is never used as an input to training and is not overwritten by reproduction.
