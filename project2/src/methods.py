"""Fixed-region estimation (equations 9–10) and the BH step-up rule."""

import numpy as np


def validate_pvalues(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("p-values must be a nonempty finite vector")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]")
    return p


def storey_fixed(p, gamma, tuning=0.5):
    p = validate_pvalues(p)
    if not 0 < gamma < 1 or not 0 <= tuning < 1:
        raise ValueError("gamma must be in (0,1), lambda in [0,1)")
    reject = p <= gamma
    pi0_raw = np.count_nonzero(p > tuning) / (len(p) * (1 - tuning))
    pi0_hat = min(1.0, pi0_raw)
    fdr_raw = pi0_hat * len(p) * gamma / max(int(reject.sum()), 1)
    # Stable even when gamma is very small.
    probability_bound = -np.expm1(len(p) * np.log1p(-gamma))
    return reject, {
        "pi0_raw": pi0_raw,
        "pi0_hat": pi0_hat,
        "fdr_hat": min(1.0, fdr_raw),
        "pfdr_hat": min(1.0, fdr_raw / probability_bound),
    }


def bh(p, alpha):
    p = validate_pvalues(p)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    order = np.argsort(p)
    ranked = p[order]
    eligible = np.flatnonzero(ranked <= alpha * np.arange(1, len(p) + 1) / len(p))
    reject = np.zeros(len(p), dtype=bool)
    if eligible.size:
        reject[order[:eligible[-1] + 1]] = True
    cutoff = float(p[reject].max()) if reject.any() else 0.0
    return reject, cutoff
