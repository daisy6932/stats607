"""Realized performance and independent analytic benchmarks."""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def rejection_metrics(reject, is_null):
    """Summarize a rejection mask against the true-null mask is_null.

    Return counts, FDP (zero for no rejections), and power (NaN for no alternatives)."""
    reject, is_null = np.asarray(reject, bool), np.asarray(is_null, bool)
    if reject.shape != is_null.shape or reject.ndim != 1:
        raise ValueError("rejections and null labels must be matching vectors")
    num_rejections = int(reject.sum())
    num_false_rejections = int(np.count_nonzero(reject & is_null))
    num_alternatives = int(np.count_nonzero(~is_null))
    return {
        "rejections": num_rejections,
        "false_rejections": num_false_rejections,
        "true_rejections": num_rejections - num_false_rejections,
        "alternatives": num_alternatives,
        "fdp": num_false_rejections / max(num_rejections, 1),
        "power": (num_rejections - num_false_rejections) / num_alternatives
        if num_alternatives
        else np.nan,
    }


def analytic_targets(num_tests, pi0, gamma, effect, mode):
    """Compute power, FDR, pFDR and the chance of any rejection.

    The design has num_tests tests, null fraction pi0, cutoff gamma and
    normal mean shift effect. Mode specifies fixed counts or iid labels."""
    power = float(norm.sf(norm.isf(gamma) - effect))
    if mode == "iid":
        rejection_probability = pi0 * gamma + (1 - pi0) * power
        probability_any_rejection = (
            -np.expm1(num_tests * np.log1p(-rejection_probability))
            if rejection_probability < 1
            else 1.0
        )
        pfdr = pi0 * gamma / rejection_probability
        fdr = pfdr * probability_any_rejection
    elif mode == "fixed":
        num_nulls = round(num_tests * pi0)
        num_alternatives = num_tests - num_nulls
        probability_any_rejection = -np.expm1(
            num_nulls * np.log1p(-gamma) + num_alternatives * np.log1p(-power)
        )
        if num_nulls:

            def integrand(integration_point):
                return np.exp(
                    (num_nulls - 1) * np.log1p(-gamma * (1 - integration_point))
                    + num_alternatives * np.log1p(-power * (1 - integration_point))
                )

            fdr = num_nulls * gamma * quad(integrand, 0, 1, epsabs=1e-12)[0]
        else:
            fdr = 0.0
        pfdr = fdr / probability_any_rejection if probability_any_rejection else np.nan
    else:
        raise ValueError("unknown mode")
    return {
        "analytic_power": power if pi0 < 1 else np.nan,
        "analytic_fdr": fdr,
        "analytic_pfdr": pfdr,
        "analytic_pr_positive": probability_any_rejection,
    }


def mean_mcse(values):
    """Summarize finite replicate values with a mean and Monte Carlo standard error.

    Return the valid count and pointwise 95% normal interval as well.
    The standard error and interval are NaN with fewer than two valid values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    num_valid = len(values)
    mean = float(values.mean()) if num_valid else np.nan
    mcse = float(values.std(ddof=1) / np.sqrt(num_valid)) if num_valid > 1 else np.nan
    return {
        "mean": mean,
        "mcse": mcse,
        "n": num_valid,
        "lower": mean - 1.96 * mcse,
        "upper": mean + 1.96 * mcse,
    }
