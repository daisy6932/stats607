"""Data generation and stable replicate-specific random streams."""

import numpy as np
from scipy.stats import norm


def replicate_rng(seed, mode, pi0, replicate):
    """Build a NumPy generator from seed, label mode, null proportion and replicate index.

    The same dataset is reused across gamma thresholds."""
    seed_components = [seed, {"fixed": 0, "iid": 1}[mode], round(1000 * pi0), replicate]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_components)))


def generate_normal_tests(rng, num_tests, pi0, effect=2.0, mode="fixed"):
    """Generate num_tests independent normal tests with alternative mean shift effect.

    Use rng and null fraction pi0; mode selects fixed counts or iid labels.
    Return upper-tail p-values and a Boolean true-null mask."""
    if not isinstance(num_tests, int) or num_tests < 1 or not 0 <= pi0 <= 1:
        raise ValueError("m must be a positive integer and pi0 in [0,1]")
    if not np.isfinite(effect):
        raise ValueError("effect must be finite")
    if mode == "fixed":
        is_null = np.arange(num_tests) < round(num_tests * pi0)
    elif mode == "iid":
        is_null = rng.random(num_tests) < pi0
    else:
        raise ValueError("mode must be fixed or iid")
    test_statistics = rng.normal(size=num_tests) + effect * (~is_null)
    return norm.sf(test_statistics), is_null
