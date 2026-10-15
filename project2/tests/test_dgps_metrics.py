import numpy as np
import pytest
from scipy.stats import binom, norm

from src.dgps import generate_normal_tests, replicate_rng
from src.metrics import analytic_targets, mean_mcse


def test_fixed_counts_and_null_pvalue_properties():
    p, null = generate_normal_tests(np.random.default_rng(11), 200000, 0.4)
    assert null.sum() == 80000
    # Generous fixed-seed distribution checks; do not assert a random sample equals its mean.
    assert abs(p[null].mean() - 0.5) < 0.005
    target = norm.sf(norm.isf(0.01)-2)
    assert abs(np.mean(p[~null] <= 0.01) - target) < 0.006


def test_iid_null_count_varies_and_boundary_labels_are_exact():
    rng = np.random.default_rng(99)
    counts = [generate_normal_tests(rng, 1000, 0.4, mode="iid")[1].sum() for _ in range(100)]
    assert len(set(counts)) > 10
    assert abs(np.mean(counts) - 400) < 5
    for mode in ("fixed", "iid"):
        for pi0 in (0, 1):
            assert generate_normal_tests(rng, 20, pi0, mode=mode)[1].sum() == 20*pi0


def test_analytic_fdr_integral_against_exact_enumeration():
    m, pi0, gamma = 6, 0.5, 0.1
    b = norm.sf(norm.isf(gamma)-2)
    fdr = sum(v / max(v+s, 1) * binom.pmf(v, 3, gamma) * binom.pmf(s, 3, b)
              for v in range(4) for s in range(4))
    targets = analytic_targets(m, pi0, gamma, 2, "fixed")
    assert targets["analytic_fdr"] == pytest.approx(fdr, abs=1e-12)


@pytest.mark.parametrize("mode", ["fixed", "iid"])
def test_all_null_fdr_and_pfdr(mode):
    t = analytic_targets(1000, 1, 0.001, 2, mode)
    assert t["analytic_fdr"] == pytest.approx(1-0.999**1000)
    assert t["analytic_pfdr"] == pytest.approx(1)
    assert np.isnan(t["analytic_power"])
    assert analytic_targets(1000, 0, 0.01, 2, mode)["analytic_fdr"] == 0


def test_mean_mcse_handles_missing_estimands():
    stats = mean_mcse([1, 2, 3, np.nan])
    assert stats["mean"] == 2
    assert stats["mcse"] == pytest.approx(1 / np.sqrt(3))
    assert stats["n"] == 3
    assert mean_mcse([np.nan])["n"] == 0


def test_random_stream_reproduction_and_independence():
    a = generate_normal_tests(replicate_rng(6072026, "fixed", 0.5, 12), 1000, 0.5)[0]
    b = generate_normal_tests(replicate_rng(6072026, "fixed", 0.5, 12), 1000, 0.5)[0]
    c = generate_normal_tests(replicate_rng(6072026, "fixed", 0.5, 13), 1000, 0.5)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
