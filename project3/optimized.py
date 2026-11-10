"""Run the Storey simulation with batched kernels and optional scenario workers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from baseline.src.dgps import replicate_rng
from baseline.src.simulation import design_groups, load_config

ROOT = Path(__file__).resolve().parent
METHODS = np.array(["storey_fixed", "bh_estimated", "bh_fixed"])
DEFAULT_BATCH_SIZE = 64


def generate_batch(config, mode, pi0, replicates):
    """Return p-values and true-null labels, with a row for each replicate index."""
    num_tests = config["m"]
    statistics = np.empty((len(replicates), num_tests))
    is_null = np.empty(statistics.shape, dtype=bool)
    fixed_labels = np.arange(num_tests) < round(num_tests * pi0)
    for row, replicate in enumerate(replicates):
        rng = replicate_rng(config["seed"], mode, pi0, replicate)
        is_null[row] = fixed_labels if mode == "fixed" else rng.random(num_tests) < pi0
        # Keep the Project 2 draw order within every replicate.
        statistics[row] = rng.normal(size=num_tests) + config["effect"] * (~is_null[row])
    return norm.sf(statistics), is_null


def bh_counts(pvalues, is_null, levels):
    """Return BH rejection counts, false counts and observed cutoffs at each level.

    The first axis indexes datasets in all inputs. Each column of levels
    specifies one level per dataset. Outputs have the shape of levels.
    """
    num_tests = pvalues.shape[1]
    ranks = np.arange(1, num_tests + 1)
    order = np.argsort(pvalues, axis=1)
    ordered_pvalues = np.take_along_axis(pvalues, order, axis=1)
    ordered_nulls = np.take_along_axis(is_null, order, axis=1)
    null_prefix = np.pad(np.cumsum(ordered_nulls, axis=1), ((0, 0), (1, 0)))
    # Preserve the baseline arithmetic order: multiply first, then divide.
    eligible = ordered_pvalues[:, None, :] <= levels[:, :, None] * ranks / num_tests
    last_rank = np.where(eligible, ranks, 0).max(axis=2)
    false_counts = np.take_along_axis(null_prefix, last_rank, axis=1)
    cutoffs = np.take_along_axis(ordered_pvalues, np.maximum(last_rank - 1, 0), axis=1)
    return last_rank, false_counts, np.where(last_rank > 0, cutoffs, 0.0)


def evaluate_batch(config, mode, pi0, role, replicates):
    """Evaluate three methods at every cutoff and pack rows in Project 2 order."""
    pvalues, is_null = generate_batch(config, mode, pi0, replicates)
    num_tests = config["m"]
    gammas = np.asarray(config["gammas"])
    num_gammas = len(gammas)
    batch_rows = len(replicates)
    lambda_threshold = config["lambda"]

    fixed_mask = pvalues[:, None, :] <= gammas[None, :, None]
    fixed_counts = fixed_mask.sum(axis=2)
    fixed_false = (fixed_mask & is_null[:, None, :]).sum(axis=2)
    pi0_raw = (pvalues > lambda_threshold).sum(axis=1) / (
        num_tests * (1 - lambda_threshold)
    )
    pi0_hat = np.minimum(1.0, pi0_raw)
    fdr_raw = pi0_hat[:, None] * num_tests * gammas / np.maximum(fixed_counts, 1)
    fdr_hat = np.minimum(1.0, fdr_raw)
    probability_bound = -np.expm1(num_tests * np.log1p(-gammas))
    pfdr_hat = np.minimum(1.0, fdr_raw / probability_bound)

    # One sort per dataset, with a single extra level for fixed-alpha BH.
    levels = np.column_stack((fdr_hat, np.full(batch_rows, config["baseline_alpha"])))
    bh_total, bh_false, bh_cutoff = bh_counts(pvalues, is_null, levels)

    def method_grid(storey, adaptive, fixed):
        """Stack the three methods on the last axis, broadcasting fixed BH over gamma."""
        shape = (batch_rows, num_gammas)
        return np.stack(
            [np.broadcast_to(values, shape) for values in (storey, adaptive, fixed)],
            axis=-1,
        )

    rejections = method_grid(fixed_counts, bh_total[:, :-1], bh_total[:, -1:])
    false_rejections = method_grid(fixed_false, bh_false[:, :-1], bh_false[:, -1:])
    cutoffs = method_grid(gammas, bh_cutoff[:, :-1], bh_cutoff[:, -1:])
    alphas = method_grid(np.nan, fdr_hat, config["baseline_alpha"])
    alternatives = (~is_null).sum(axis=1)[:, None, None]
    true_rejections = rejections - false_rejections
    power = np.full(rejections.shape, np.nan)
    np.divide(true_rejections, alternatives, out=power, where=alternatives > 0)

    def flatten(values):
        """Broadcast metadata or metrics to replicate × gamma × method row order."""
        return np.broadcast_to(values, rejections.shape).ravel()

    return pd.DataFrame(
        {
            "mode": mode,
            "role": role,
            "pi0": pi0,
            "gamma": flatten(gammas[:, None]),
            "replicate": flatten(np.asarray(replicates)[:, None, None]),
            "seed": config["seed"],
            "m": num_tests,
            "effect": config["effect"],
            "lambda": lambda_threshold,
            "method": flatten(METHODS),
            "cutoff": flatten(cutoffs),
            "alpha": flatten(alphas),
            "pi0_raw": flatten(pi0_raw[:, None, None]),
            "pi0_hat": flatten(pi0_hat[:, None, None]),
            "fdr_hat": flatten(fdr_hat[:, :, None]),
            "pfdr_hat": flatten(pfdr_hat[:, :, None]),
            "rejections": flatten(rejections),
            "false_rejections": flatten(false_rejections),
            "true_rejections": flatten(true_rejections),
            "alternatives": flatten(alternatives),
            "fdp": flatten(false_rejections / np.maximum(rejections, 1)),
            "power": flatten(power),
        }
    )


def run_group(task):
    """Compute one design group's batches and return its ordered rows."""
    config, batch_size, (mode, pi0, role) = task
    total = config["replicates"]
    return pd.concat(
        [
            evaluate_batch(
                config, mode, pi0, role, range(start, min(start + batch_size, total))
            )
            for start in range(0, total, batch_size)
        ],
        ignore_index=True,
    )


def run_study(config, workers=1, batch_size=DEFAULT_BATCH_SIZE):
    """Return the study rows in design order, independent of worker and batch counts."""
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch size must be positive")
    tasks = [(config, batch_size, group) for group in design_groups(config)]
    if workers == 1:
        groups = list(map(run_group, tasks))
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            groups = list(pool.map(run_group, tasks))
    return pd.concat(groups, ignore_index=True)


def main():
    """Run a fresh simulation and write the raw CSV and reproducibility metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/study.json"))
    parser.add_argument("--output", type=Path, default=Path("tmp/study"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    config = load_config(args.config)
    start = time.perf_counter()
    raw = run_study(config, args.workers, args.batch_size)
    directory = args.output / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "replicates.csv.gz"
    temporary = directory / "replicates.tmp.csv.gz"
    raw.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    temporary.replace(archive)
    metadata = {
        "config": config,
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "pandas", "matplotlib")
        },
        "source_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for source in (
                "optimized.py",
                "baseline/src/dgps.py",
                "baseline/src/simulation.py",
            )
        },
        "seconds": time.perf_counter() - start,
        "rows": len(raw),
        "raw_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(raw):,} rows in {metadata['seconds']:.2f} s")


if __name__ == "__main__":
    main()
