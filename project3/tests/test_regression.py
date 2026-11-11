"""Compare optimized, parallel and baseline results under the same random streams."""

import numpy as np
import pandas as pd
import pytest

from baseline.src.analyze import analyze
from baseline.src.methods import bh
from baseline.src.simulation import load_config, run_study as baseline_study
from optimized import bh_counts, run_study


@pytest.fixture(scope="module")
def study_config():
    """Load the complete, unchanged Project 2 design."""
    return load_config("config/study.json")


def test_full_study_rows_match_exactly(study_config):
    """Check all 120,000 rows, including cutoffs, counts and estimators."""
    pd.testing.assert_frame_equal(
        baseline_study(study_config), run_study(study_config), check_exact=True
    )


def test_parallel_and_partial_batches_match(study_config):
    """Exercise spawn workers, a partial batch, and thresholds in a different order."""
    config = dict(study_config, replicates=11, pi0_values=[0.8], gammas=[0.001, 0.01])
    expected = baseline_study(config)
    pd.testing.assert_frame_equal(
        expected, run_study(config, workers=2, batch_size=4), check_exact=True
    )
    pd.testing.assert_frame_equal(
        expected, run_study(config, batch_size=1), check_exact=True
    )


def test_bh_ties_zero_and_full_rejections():
    """Compare batched BH against the baseline masks on tied and extreme inputs."""
    pvalues = np.array([[0.04, 0.04, 0.9, 1.0], [0.0, 0.04, 0.9, 1.0]])
    null_labels = np.array([[True, False, True, True], [False, True, True, False]])
    levels = np.array([[0.0, 0.08, 1.0], [0.0, 0.08, 1.0]])
    counts, false_counts, cutoffs = bh_counts(pvalues, null_labels, levels)
    for row in range(len(pvalues)):
        for column, level in enumerate(levels[row]):
            mask, cutoff = bh(pvalues[row], level)
            assert counts[row, column] == mask.sum()
            assert false_counts[row, column] == (mask & null_labels[row]).sum()
            assert cutoffs[row, column] == cutoff


def test_no_alternatives_and_small_thresholds(study_config):
    """Preserve NaN power for all nulls and zero false discoveries for all signals."""
    config = dict(
        study_config, replicates=9, pi0_values=[0.8], gammas=[1e-12, 1e-8], effect=12.0
    )
    result = run_study(config, batch_size=4)
    pd.testing.assert_frame_equal(baseline_study(config), result, check_exact=True)
    all_null = result[result.pi0 == 1]
    all_alternative = result[result.pi0 == 0]
    assert all_null.power.isna().all()
    assert (all_alternative.false_rejections == 0).all()
    assert np.isfinite(result.pfdr_hat).all()


def test_summary_tables_match(study_config, tmp_path):
    """Pass both engines through the unchanged analysis and compare every output table."""
    config = dict(study_config, replicates=17, pi0_values=[0.2, 0.8])
    reference = "baseline/results/reference/table2.csv"
    baseline_dir, optimized_dir = tmp_path / "baseline", tmp_path / "optimized"
    analyze(baseline_study(config), config, baseline_dir, reference)
    analyze(run_study(config), config, optimized_dir, reference)
    for file in baseline_dir.glob("*.csv"):
        assert file.read_bytes() == (optimized_dir / file.name).read_bytes(), file.name


def test_cli_writes_raw_and_metadata(study_config, tmp_path):
    """Exercise the command-line serialization path and verify the recorded archive hash."""
    import hashlib
    import json
    import subprocess
    import sys

    config = dict(study_config, replicates=3, pi0_values=[0.8])
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "optimized",
            "--config",
            str(path),
            "--output",
            str(tmp_path),
        ],
        check=True,
    )
    archive = tmp_path / "raw/replicates.csv.gz"
    metadata = json.loads((archive.parent / "metadata.json").read_text())
    assert metadata["raw_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert metadata["rows"] == len(baseline_study(config))
    assert set(metadata["source_sha256"]) == {
        "optimized.py",
        "baseline/src/dgps.py",
        "baseline/src/simulation.py",
    }


def test_complexity_fit_uses_only_large_sizes(tmp_path):
    """Small-size overhead must not affect the reported tail slope."""
    from performance import fit_complexity

    sizes = [250, 500, 1000, 2000, 4000, 8000]
    data = pd.DataFrame(
        {
            "engine": ["baseline"] * 6,
            "m": sizes,
            "seconds": [9, 8, 7, 2, 4, 8],
        }
    )
    report = fit_complexity(data, tmp_path)
    assert report["engines"]["baseline"]["sizes"] == [2000, 4000, 8000]
    assert report["engines"]["baseline"]["slope"] == pytest.approx(1)
    data.loc[data.m < 2000, "seconds"] *= 100
    updated = fit_complexity(data, tmp_path)
    assert updated["engines"]["baseline"]["slope"] == report["engines"]["baseline"]["slope"]
    assert (
        updated["engines"]["baseline"]["full_range_slope"]
        != report["engines"]["baseline"]["full_range_slope"]
    )
