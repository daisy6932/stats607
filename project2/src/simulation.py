"""Run every configured replicate and save raw output before summarization."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dgps import generate_normal_tests, replicate_rng
from .methods import bh, storey_fixed
from .metrics import rejection_metrics

LOG = logging.getLogger(__name__)


def load_config(path):
    """Read and validate the study JSON at path; return its configuration dictionary."""
    config = json.loads(Path(path).read_text())
    for field in ("m", "replicates"):
        if not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(config["effect"]):
        raise ValueError("effect must be finite")
    for field in ("pi0_values", "boundary_pi0"):
        values = config[field]
        if len(values) != len(set(values)) or any(not 0 <= pi0 <= 1 for pi0 in values):
            raise ValueError(f"invalid or duplicate {field}")
        if any(not np.isclose(1000 * pi0, round(1000 * pi0)) for pi0 in values):
            raise ValueError("pi0 seed keys require a grid with resolution 0.001")
    if not config["pi0_values"] or not config["modes"] or not config["gammas"]:
        raise ValueError("design axes must be nonempty")
    if set(config["pi0_values"]) & set(config["boundary_pi0"]):
        raise ValueError("boundary and primary pi0 values must not overlap")
    if len(config["modes"]) != len(set(config["modes"])) or not set(config["modes"]) <= {
        "fixed",
        "iid",
    }:
        raise ValueError("invalid modes")
    if len(config["gammas"]) != len(set(config["gammas"])) or any(
        not 0 < gamma < 1 for gamma in config["gammas"]
    ):
        raise ValueError("invalid gamma values")
    if not 0 <= config["lambda"] < 1 or not 0 <= config["baseline_alpha"] <= 1:
        raise ValueError("invalid method parameters")
    return config


def design_groups(config):
    """Yield (label mode, null fraction, analysis role) triples from the configuration."""
    for mode in config["modes"]:
        for pi0 in config["pi0_values"]:
            yield mode, pi0, "primary" if mode == "fixed" else "sensitivity"
    for pi0 in config["boundary_pi0"]:
        yield "fixed", pi0, "boundary"


def run_study(config):
    """Simulate the conditions in config and return a DataFrame.

    Each row represents one replicate, rejection threshold and method."""
    rows = []
    for mode, pi0, role in design_groups(config):
        LOG.info("%s pi0=%.3f: %d replicates", mode, pi0, config["replicates"])
        for replicate in range(config["replicates"]):
            try:
                rng = replicate_rng(config["seed"], mode, pi0, replicate)
                pvalues, is_null = generate_normal_tests(
                    rng, config["m"], pi0, config["effect"], mode
                )
                for gamma in config["gammas"]:
                    storey_reject, estimates = storey_fixed(
                        pvalues, gamma, config["lambda"]
                    )
                    bh_at_estimated_fdr, estimated_cutoff = bh(
                        pvalues, estimates["fdr_hat"]
                    )
                    bh_at_fixed_level, baseline_cutoff = bh(
                        pvalues, config["baseline_alpha"]
                    )
                    for method, reject, cutoff, alpha in (
                        ("storey_fixed", storey_reject, gamma, np.nan),
                        (
                            "bh_estimated",
                            bh_at_estimated_fdr,
                            estimated_cutoff,
                            estimates["fdr_hat"],
                        ),
                        (
                            "bh_fixed",
                            bh_at_fixed_level,
                            baseline_cutoff,
                            config["baseline_alpha"],
                        ),
                    ):
                        rows.append(
                            {
                                "mode": mode,
                                "role": role,
                                "pi0": pi0,
                                "gamma": gamma,
                                "replicate": replicate,
                                "seed": config["seed"],
                                "m": config["m"],
                                "effect": config["effect"],
                                "lambda": config["lambda"],
                                "method": method,
                                "cutoff": cutoff,
                                "alpha": alpha,
                                **estimates,
                                **rejection_metrics(reject, is_null),
                            }
                        )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed: mode={mode}, pi0={pi0}, replicate={replicate}"
                ) from exc
    return pd.DataFrame(rows)


def main():
    """Parse command-line options and write or reuse the raw simulation results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.json")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if the verified cache matches"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    out = Path(args.output) / "raw"
    path = out / "replicates.csv.gz"
    metadata_path = out / "metadata.json"
    hashes = {
        str(source_path): hashlib.sha256(source_path.read_bytes()).hexdigest()
        for source_path in sorted(Path("src").glob("*.py"))
    }
    versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy", "pandas", "matplotlib", "pytest")
    }
    if not args.force and path.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (
            previous["config"] == config
            and previous["source_sha256"] == hashes
            and previous["versions"] == versions
            and previous["python"] == platform.python_version()
            and previous["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            LOG.info("Reusing verified raw cache: %s", path)
            return
    start = time.perf_counter()
    raw = run_study(config)
    out.mkdir(parents=True, exist_ok=True)
    temp = out / "replicates.tmp.csv.gz"
    raw.to_csv(
        temp, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0}
    )
    temp.replace(path)
    metadata = {
        "config": config,
        "config_path": args.config,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "source_sha256": hashes,
        "python": platform.python_version(),
        "versions": versions,
        "seconds": time.perf_counter() - start,
        "rows": len(raw),
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOG.info("Saved %d raw rows in %.2f s", len(raw), metadata["seconds"])


if __name__ == "__main__":
    main()
