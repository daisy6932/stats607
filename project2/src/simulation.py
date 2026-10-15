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
    cfg = json.loads(Path(path).read_text())
    for field in ("m", "replicates"):
        if not isinstance(cfg[field], int) or cfg[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(cfg["effect"]):
        raise ValueError("effect must be finite")
    for field in ("pi0_values", "boundary_pi0"):
        values = cfg[field]
        if len(values) != len(set(values)) or any(not 0 <= p <= 1 for p in values):
            raise ValueError(f"invalid or duplicate {field}")
        if any(not np.isclose(1000 * p, round(1000 * p)) for p in values):
            raise ValueError("pi0 seed keys require a grid with resolution 0.001")
    if not cfg["pi0_values"] or not cfg["modes"] or not cfg["gammas"]:
        raise ValueError("design axes must be nonempty")
    if set(cfg["pi0_values"]) & set(cfg["boundary_pi0"]):
        raise ValueError("boundary and primary pi0 values must not overlap")
    if len(cfg["modes"]) != len(set(cfg["modes"])) or not set(cfg["modes"]) <= {"fixed", "iid"}:
        raise ValueError("invalid modes")
    if len(cfg["gammas"]) != len(set(cfg["gammas"])) or any(not 0 < g < 1 for g in cfg["gammas"]):
        raise ValueError("invalid gamma values")
    if not 0 <= cfg["lambda"] < 1 or not 0 <= cfg["baseline_alpha"] <= 1:
        raise ValueError("invalid method parameters")
    return cfg


def design_groups(cfg):
    for mode in cfg["modes"]:
        for pi0 in cfg["pi0_values"]:
            yield mode, pi0, "primary" if mode == "fixed" else "sensitivity"
    for pi0 in cfg["boundary_pi0"]:
        yield "fixed", pi0, "boundary"


def run_study(cfg):
    rows = []
    for mode, pi0, role in design_groups(cfg):
        LOG.info("%s pi0=%.3f: %d replicates", mode, pi0, cfg["replicates"])
        for rep in range(cfg["replicates"]):
            try:
                rng = replicate_rng(cfg["seed"], mode, pi0, rep)
                p, is_null = generate_normal_tests(rng, cfg["m"], pi0, cfg["effect"], mode)
                for gamma in cfg["gammas"]:
                    fixed, estimates = storey_fixed(p, gamma, cfg["lambda"])
                    matched, cutoff = bh(p, estimates["fdr_hat"])
                    baseline, baseline_cutoff = bh(p, cfg["baseline_alpha"])
                    for method, reject, cut, alpha in (
                        ("storey_fixed", fixed, gamma, np.nan),
                        ("bh_estimated", matched, cutoff, estimates["fdr_hat"]),
                        ("bh_fixed", baseline, baseline_cutoff, cfg["baseline_alpha"]),
                    ):
                        rows.append({"mode": mode, "role": role, "pi0": pi0, "gamma": gamma,
                                     "replicate": rep, "seed": cfg["seed"], "m": cfg["m"],
                                     "effect": cfg["effect"], "lambda": cfg["lambda"],
                                     "method": method, "cutoff": cut, "alpha": alpha,
                                     **estimates, **rejection_metrics(reject, is_null)})
            except Exception as exc:
                raise RuntimeError(f"Failed: mode={mode}, pi0={pi0}, replicate={rep}") from exc
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the verified cache matches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    out = Path(args.output) / "raw"
    path = out / "replicates.csv.gz"
    metadata_path = out / "metadata.json"
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path("src").glob("*.py"))}
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "scipy", "pandas", "matplotlib", "pytest")}
    if not args.force and path.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (previous["config"] == cfg and previous["source_sha256"] == hashes
                and previous["versions"] == versions
                and previous["python"] == platform.python_version()
                and previous["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()):
            LOG.info("Reusing verified raw cache: %s", path)
            return
    start = time.perf_counter()
    raw = run_study(cfg)
    out.mkdir(parents=True, exist_ok=True)
    temp = out / "replicates.tmp.csv.gz"
    raw.to_csv(temp, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    temp.replace(path)
    metadata = {"config": cfg, "config_path": args.config,
                "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "source_sha256": hashes, "python": platform.python_version(),
                "versions": versions,
                "seconds": time.perf_counter() - start, "rows": len(raw),
                "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOG.info("Saved %d raw rows in %.2f s", len(raw), metadata["seconds"])


if __name__ == "__main__":
    main()
