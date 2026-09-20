"""Data, policy-value, failure-path, and complete-pipeline validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from its_compare_value_first_2_grouped.data import generate_synthetic_data_with_truth
from its_compare_value_first_2_grouped.eval_policy import evaluate_policy
from pipeline import reproduce, validate_results


def test_known_policy_values_and_tied_optimal_actions():
    means = np.array([[1., 3., 3.], [4., 1., 0.]])
    result = evaluate_policy(means, np.array([0, 0]), np.array([1, 0]))
    assert result == {"value": 2.5, "oracle_value": 3.5, "regret": 1., "acc_opt": .5}
    tied = evaluate_policy(means, np.array([2, 0]), np.array([1, 0]))
    assert tied["regret"] == 0 and tied["acc_opt"] == 1


def test_synthetic_data_shapes_ranges_and_seed_reproducibility():
    kwargs = dict(n_samples=80, n_features=10, n_treatments=10,
                  groups=[list(range(5)), list(range(5, 10))], seed=607,
                  scenario="grouped_nonlinear_hard")
    first = generate_synthetic_data_with_truth(**kwargs)
    second = generate_synthetic_data_with_truth(**kwargs)
    assert first["X"].shape == (80, 10)
    assert first["mu_true"].shape == (80, 10)
    for key in first:
        assert np.isfinite(first[key].numpy()).all()
        np.testing.assert_array_equal(first[key].numpy(), second[key].numpy())
    assert first["A"].min() >= 0 and first["A"].max() < 10
    means = first["mu_true"].numpy()
    optimal = first["a_opt"].numpy()
    np.testing.assert_allclose(means[np.arange(80), optimal], means.max(axis=1))


def test_validation_rejects_incomplete_or_inconsistent_results():
    cfg = {"seed0": 1, "n_reps": 1, "methods": "ridge"}
    row = dict(seed=1, method="ridge", regret=.5, value=1., oracle_value=1.5, acc_opt=.8)
    validate_results(pd.DataFrame([row]), cfg)
    with pytest.raises(ValueError, match="duplicate"):
        validate_results(pd.DataFrame([row, row]), cfg)
    with pytest.raises(ValueError, match="Regret must equal"):
        validate_results(pd.DataFrame([{**row, "regret": .1}]), cfg)
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_results(pd.DataFrame([{**row, "value": np.nan}]), cfg)


def test_pipeline_from_scratch(tmp_path):
    reproduce(tmp_path)
    table = pd.read_csv(tmp_path / "results/tables/policy_comparison.csv")
    assert set(table.method) == {"value_first_nn", "initial_nn", "ridge"}
    assert (table.n == 3).all()
    assert len(list((tmp_path / "data/processed").glob("*.npz"))) == 9
    assert (tmp_path / "results/figures/policy_regret.png").stat().st_size > 1000
    assert (tmp_path / "results/report/summary.md").is_file()
