"""Summarize cached replicates, uncertainty, and published reference differences."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .metrics import analytic_targets, mean_mcse
from .simulation import design_groups

KEYS = ["mode", "role", "pi0", "gamma"]


def validate_raw(raw, config):
    """Check replicate rows against config and raise ValueError on inconsistent output.

    Check condition keys, method coverage, counts and the FDP and power formulas."""
    expected_keys = {
        (mode, role, pi0, gamma)
        for mode, pi0, role in design_groups(config)
        for gamma in config["gammas"]
    }
    observed_keys = set(raw[KEYS].itertuples(index=False, name=None))
    if observed_keys != expected_keys:
        raise ValueError("raw conditions do not match configuration")
    if set(raw.method) != {"storey_fixed", "bh_estimated", "bh_fixed"}:
        raise ValueError("missing or unknown methods")
    if raw.duplicated(KEYS + ["method", "replicate"]).any():
        raise ValueError("duplicate replicate keys")
    if len(raw) != len(expected_keys) * 3 * config["replicates"]:
        raise ValueError("incomplete raw output")
    for _, group in raw.groupby(KEYS + ["method"]):
        if set(group.replicate) != set(range(config["replicates"])):
            raise ValueError("incomplete replicate group")
    for col in ("fdp", "cutoff", "pi0_hat", "fdr_hat", "pfdr_hat"):
        if not raw[col].between(0, 1).all():
            raise ValueError(f"invalid {col}")
    for col in ("m", "seed", "effect", "lambda"):
        if not (raw[col] == config[col]).all():
            raise ValueError(f"raw/config mismatch: {col}")
    counts = raw[["rejections", "false_rejections", "true_rejections", "alternatives"]]
    if (
        counts.isna().any().any()
        or (counts < 0).any().any()
        or (counts > config["m"]).any().any()
    ):
        raise ValueError("invalid counts")
    if not np.equal(counts, np.floor(counts)).all().all():
        raise ValueError("counts must be integers")
    if not (raw.rejections == raw.false_rejections + raw.true_rejections).all():
        raise ValueError("inconsistent counts")
    if (raw.true_rejections > raw.alternatives).any() or (
        raw.false_rejections > config["m"] - raw.alternatives
    ).any():
        raise ValueError("rejections exceed available hypotheses")
    if not np.allclose(raw.fdp, raw.false_rejections / raw.rejections.clip(lower=1)):
        raise ValueError("inconsistent FDP")
    has_alternatives = raw.alternatives > 0
    if not np.allclose(
        raw.loc[has_alternatives, "power"],
        raw.loc[has_alternatives, "true_rejections"]
        / raw.loc[has_alternatives, "alternatives"],
    ):
        raise ValueError("inconsistent power")
    if not raw.loc[~has_alternatives, "power"].isna().all():
        raise ValueError("power must be undefined without alternatives")


def analyze(raw, config, output_dir, reference_path):
    """Summarize raw replicate rows according to config.

    Read the published Table 2 CSV at reference_path and write summary CSVs
    and paired comparisons into output_dir."""
    validate_raw(raw, config)
    summary, diagnostics, differences = [], [], []
    for key, group in raw.groupby(KEYS, sort=False):
        condition = dict(zip(KEYS, key))
        target = analytic_targets(
            config["m"],
            condition["pi0"],
            condition["gamma"],
            config["effect"],
            condition["mode"],
        )
        for method, values in group.groupby("method"):
            for metric in ("power", "fdp", "cutoff", "rejections"):
                summary.append(
                    {
                        **condition,
                        "method": method,
                        "metric": metric,
                        **mean_mcse(values[metric]),
                        **target,
                    }
                )
            summary.append(
                {
                    **condition,
                    "method": method,
                    "metric": "pfdr",
                    **mean_mcse(values.loc[values.rejections > 0, "fdp"]),
                    **target,
                }
            )
        fixed = group[group.method == "storey_fixed"].set_index("replicate")
        for estimate, truth in (
            ("pi0_hat", condition["pi0"]),
            ("pi0_raw", condition["pi0"]),
            ("fdr_hat", target["analytic_fdr"]),
            ("pfdr_hat", target["analytic_pfdr"]),
        ):
            errors = fixed[estimate] - truth
            diagnostics.append(
                {
                    **condition,
                    "estimate": estimate,
                    "truth": truth,
                    "estimate_mean": float(fixed[estimate].mean()),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    **mean_mcse(errors),
                    "pi0_cap_frequency": float((fixed.pi0_raw > 1).mean()),
                }
            )
        for comparator in ("bh_estimated", "bh_fixed"):
            bh_rows = group[group.method == comparator].set_index("replicate")
            differences.append(
                {
                    **condition,
                    "comparison": f"storey_fixed - {comparator}",
                    **mean_mcse(fixed.power - bh_rows.power),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics.csv", index=False)
    pd.DataFrame(differences).to_csv(output_dir / "paired_power.csv", index=False)
    design = raw[KEYS + ["m", "effect", "lambda"]].drop_duplicates()
    design["replicates"] = config["replicates"]
    design.to_csv(output_dir / "design_matrix.csv", index=False)

    # Compare each reported column against Table 2.
    paper_table = pd.read_csv(reference_path)
    comparisons = []
    mapping = {
        "paper_fdr": ("storey_fixed", "fdp"),
        "paper_storey_power": ("storey_fixed", "power"),
        "paper_bh_power": ("bh_estimated", "power"),
        "paper_fdr_hat": ("storey_fixed", "fdr_hat"),
        "paper_pi0_hat": ("storey_fixed", "pi0_hat"),
        "paper_bh_cutoff": ("bh_estimated", "cutoff"),
    }
    for paper_row in paper_table.to_dict("records"):
        condition_rows = raw[
            (raw.role == "primary")
            & (raw.pi0 == paper_row["pi0"])
            & (raw.gamma == paper_row["gamma"])
        ]
        if condition_rows.empty:
            continue
        for column, (method, metric) in mapping.items():
            stats = mean_mcse(condition_rows.loc[condition_rows.method == method, metric])
            comparisons.append(
                {
                    "pi0": paper_row["pi0"],
                    "gamma": paper_row["gamma"],
                    "column": column,
                    "paper": paper_row[column],
                    **stats,
                    "difference": stats["mean"] - paper_row[column],
                }
            )
    pd.DataFrame(comparisons).to_csv(output_dir / "table2_comparison.csv", index=False)
    # Closed-form expectation before clipping: useful to explain pi0 overestimation.
    expected_pi0 = [
        {
            "pi0": pi0,
            "expected_pi0_raw": pi0
            + (1 - pi0)
            * norm.cdf(norm.isf(config["lambda"]) - config["effect"])
            / (1 - config["lambda"]),
        }
        for pi0 in sorted(raw.pi0.unique())
    ]
    pd.DataFrame(expected_pi0).to_csv(output_dir / "pi0_expectation.csv", index=False)


def main():
    """Parse options, verify the raw file checksum and write the analysis tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--reference", default="results/reference/table2.csv")
    args = parser.parse_args()
    output_dir = Path(args.output)
    metadata = json.loads((output_dir / "raw/metadata.json").read_text())
    config = metadata["config"]
    path = output_dir / "raw/replicates.csv.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["raw_sha256"]:
        raise ValueError("raw file hash does not match metadata; regenerate the simulation")
    # pandas 3.1 uses round-trip precision by default.
    pandas_version = tuple(int(part) for part in pd.__version__.split(".")[:2])
    csv_options = {"float_precision": "round_trip"} if pandas_version < (3, 1) else {}
    raw = pd.read_csv(path, **csv_options)
    analyze(raw, config, output_dir, args.reference)
    print(f"Validated and summarized {len(raw):,} raw rows in {output_dir}")


if __name__ == "__main__":
    main()
