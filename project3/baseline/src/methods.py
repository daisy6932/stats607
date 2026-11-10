"""Fixed-region estimation (equations 9–10) and the BH step-up rule."""

import numpy as np


def validate_pvalues(pvalues):
    """Convert pvalues to a finite one-dimensional float array in [0, 1].

    Raise ValueError for an empty vector or invalid values."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or pvalues.size == 0 or not np.all(np.isfinite(pvalues)):
        raise ValueError("p-values must be a nonempty finite vector")
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must lie in [0,1]")
    return pvalues


def storey_fixed(pvalues, gamma, lambda_threshold=0.5):
    """Estimate FDR and pFDR for the fixed rejection region p <= gamma.

    The lambda_threshold determines which p-values estimate the null fraction.
    Return the rejection mask and a dictionary with raw and capped null-fraction
    estimates, FDR, and pFDR. The latter two estimates are capped at one."""
    pvalues = validate_pvalues(pvalues)
    if not 0 < gamma < 1 or not 0 <= lambda_threshold < 1:
        raise ValueError("gamma must be in (0,1), lambda in [0,1)")
    rejection_mask = pvalues <= gamma
    pi0_raw = np.count_nonzero(pvalues > lambda_threshold) / (
        len(pvalues) * (1 - lambda_threshold)
    )
    pi0_hat = min(1.0, pi0_raw)
    fdr_raw = pi0_hat * len(pvalues) * gamma / max(int(rejection_mask.sum()), 1)
    # Stable even when gamma is very small.
    probability_bound = -np.expm1(len(pvalues) * np.log1p(-gamma))
    return rejection_mask, {
        "pi0_raw": pi0_raw,
        "pi0_hat": pi0_hat,
        "fdr_hat": min(1.0, fdr_raw),
        "pfdr_hat": min(1.0, fdr_raw / probability_bound),
    }


def bh(pvalues, alpha):
    """Apply Benjamini–Hochberg at level alpha to a vector of p-values.

    Return a Boolean mask in input order and the largest rejected observed
    p-value, with cutoff zero when the mask contains no rejections."""
    pvalues = validate_pvalues(pvalues)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    order = np.argsort(pvalues)
    sorted_pvalues = pvalues[order]
    qualifying_indices = np.flatnonzero(
        sorted_pvalues <= alpha * np.arange(1, len(pvalues) + 1) / len(pvalues)
    )
    rejection_mask = np.zeros(len(pvalues), dtype=bool)
    if qualifying_indices.size:
        rejection_mask[order[: qualifying_indices[-1] + 1]] = True
    cutoff = float(pvalues[rejection_mask].max()) if rejection_mask.any() else 0.0
    return rejection_mask, cutoff
