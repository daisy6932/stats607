"""Confirm that reference checks tolerate row order but detect changed results."""
import pandas as pd
import pytest
from compare_results import ROOT, compare


def test_reference_comparison_accepts_reordered_rows(tmp_path):
    reference = ROOT / "results/reference/policy_comparison.csv"
    frame = pd.read_csv(reference).iloc[::-1]
    actual = tmp_path / "actual.csv"
    frame.to_csv(actual, index=False)
    compare(actual, reference)


def test_reference_comparison_rejects_changed_value(tmp_path):
    reference = ROOT / "results/reference/policy_comparison.csv"
    frame = pd.read_csv(reference)
    frame.loc[0, "mean_regret"] += .1
    actual = tmp_path / "actual.csv"
    frame.to_csv(actual, index=False)
    with pytest.raises(AssertionError):
        compare(actual, reference)
