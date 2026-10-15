import numpy as np
import pytest

from src.methods import bh, storey_fixed
from src.metrics import rejection_metrics


def test_bh_step_up_continues_after_earlier_failure():
    # The first ordered p misses its rank-1 threshold, but rank 3 qualifies.
    reject, cutoff = bh([0.04, 0.9, 0.02, 0.03], 0.06)
    assert reject.tolist() == [True, False, True, True]
    assert cutoff == 0.04


@pytest.mark.parametrize("p,alpha,expected", [([0.5, 0.9], 0.05, [False, False]),
    ([0.01, 0.01, 0.9], 0.05, [True, True, False]),
    ([0.1, 1.0], 1.0, [True, True]), ([0.0, 0.5], 0.0, [True, False])])
def test_bh_boundaries(p, alpha, expected):
    assert bh(p, alpha)[0].tolist() == expected


def test_bh_agrees_with_exhaustive_definition_and_permutation():
    rng = np.random.default_rng(77)
    for _ in range(20):
        p = rng.uniform(size=25)
        alpha = rng.uniform()
        ordered = sorted(p)
        k = max([0] + [i for i, x in enumerate(ordered, 1) if x <= alpha * i / len(p)])
        expected = p <= ordered[k-1] if k else np.zeros(len(p), bool)
        assert np.array_equal(bh(p, alpha)[0], expected)
        perm = rng.permutation(len(p))
        assert np.array_equal(bh(p[perm], alpha)[0], expected[perm])


def test_storey_arithmetic_and_distinct_pfdr_correction():
    reject, estimates = storey_fixed([0.001, 0.009, 0.4, 0.8], 0.01)
    assert reject.tolist() == [True, True, False, False]
    assert estimates["pi0_hat"] == 0.5
    assert estimates["fdr_hat"] == pytest.approx(0.01)
    assert estimates["pfdr_hat"] == pytest.approx(0.01 / (1-0.99**4))


def test_zero_rejections_caps_and_undefined_power():
    rejected, estimates = storey_fixed([0.8, 0.9], 0.01)
    assert not rejected.any()
    assert estimates["pi0_raw"] == 2
    assert estimates["pi0_hat"] == 1
    assert estimates["fdr_hat"] == 0.02
    assert estimates["pfdr_hat"] == 1
    metrics = rejection_metrics(rejected, [True, True])
    assert metrics["fdp"] == 0
    assert np.isnan(metrics["power"])


@pytest.mark.parametrize("p", [[], [-0.1], [1.01], [np.nan], [[0.1]]])
def test_invalid_pvalues_fail_loudly(p):
    with pytest.raises(ValueError):
        bh(p, 0.05)


def test_metrics_known_counts():
    m = rejection_metrics([True, True, False, True], [True, False, False, False])
    assert m["false_rejections"] == 1
    assert m["rejections"] == 3
    assert m["fdp"] == pytest.approx(1/3)
    assert m["power"] == pytest.approx(2/3)
