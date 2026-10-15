"""Realized performance and independent analytic benchmarks."""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def rejection_metrics(reject, is_null):
    reject, is_null = np.asarray(reject, bool), np.asarray(is_null, bool)
    if reject.shape != is_null.shape or reject.ndim != 1:
        raise ValueError("rejections and null labels must be matching vectors")
    r = int(reject.sum())
    v = int(np.count_nonzero(reject & is_null))
    m1 = int(np.count_nonzero(~is_null))
    return {"rejections": r, "false_rejections": v, "true_rejections": r - v,
            "alternatives": m1, "fdp": v / max(r, 1),
            "power": (r - v) / m1 if m1 else np.nan}


def analytic_targets(m, pi0, gamma, effect, mode):
    power = float(norm.sf(norm.isf(gamma) - effect))
    if mode == "iid":
        q = pi0 * gamma + (1 - pi0) * power
        positive = -np.expm1(m * np.log1p(-q)) if q < 1 else 1.0
        pfdr = pi0 * gamma / q
        fdr = pfdr * positive
    elif mode == "fixed":
        m0 = round(m * pi0)
        m1 = m - m0
        positive = -np.expm1(m0 * np.log1p(-gamma) + m1 * np.log1p(-power))
        if m0:
            def integrand(t):
                return np.exp((m0 - 1) * np.log1p(-gamma * (1 - t))
                              + m1 * np.log1p(-power * (1 - t)))
            fdr = m0 * gamma * quad(integrand, 0, 1, epsabs=1e-12)[0]
        else:
            fdr = 0.0
        pfdr = fdr / positive if positive else np.nan
    else:
        raise ValueError("unknown mode")
    return {"analytic_power": power if pi0 < 1 else np.nan,
            "analytic_fdr": fdr, "analytic_pfdr": pfdr,
            "analytic_pr_positive": positive}


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {"mean": mean, "mcse": se, "n": n,
            "lower": mean - 1.96 * se, "upper": mean + 1.96 * se}
