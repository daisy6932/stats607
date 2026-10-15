"""Data generation and stable replicate-specific random streams."""

import numpy as np
from scipy.stats import norm


def replicate_rng(seed, mode, pi0, replicate):
    """The two gamma conditions intentionally share each generated dataset."""
    key = [seed, {"fixed": 0, "iid": 1}[mode], round(1000 * pi0), replicate]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))


def generate_normal_tests(rng, m, pi0, effect=2.0, mode="fixed"):
    if not isinstance(m, int) or m < 1 or not 0 <= pi0 <= 1:
        raise ValueError("m must be a positive integer and pi0 in [0,1]")
    if not np.isfinite(effect):
        raise ValueError("effect must be finite")
    if mode == "fixed":
        is_null = np.arange(m) < round(m * pi0)
    elif mode == "iid":
        is_null = rng.random(m) < pi0
    else:
        raise ValueError("mode must be fixed or iid")
    z = rng.normal(size=m) + effect * (~is_null)
    return norm.sf(z), is_null
