"""Profile the Storey study and time identical inputs without reading cached output."""

import argparse
import cProfile
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
import time
import warnings

import pandas as pd

from baseline.src.simulation import load_config, run_study as baseline_study

ROOT = Path(__file__).resolve().parent
BASELINE_COMMIT = "b5feef94ab3b6dbe2f0fec3283cf08ef40b5f586"
SIZES = (250, 500, 1000, 2000, 4000, 8000)
WORKERS = (1, 2, 4)


def timed_study(engine, config, workers=1):
    """Time generation and evaluation in memory; return elapsed seconds and output."""
    start = time.perf_counter()
    if engine == "baseline":
        result = baseline_study(config)
    else:
        optimized = importlib.import_module("optimized")
        result = optimized.run_study(config, workers=workers)
    return time.perf_counter() - start, result


def run_full(engine, config_path, workers):
    """Time the three Project 2 workflow stages in fresh subprocesses and output."""
    timings = {}
    with tempfile.TemporaryDirectory(prefix="storey-benchmark-") as temporary:
        for stage in ("simulate", "analyze", "figures"):
            module = {
                "simulate": "src.simulation",
                "analyze": "src.analyze",
                "figures": "src.figures",
            }[stage]
            cwd = ROOT / "baseline"
            if engine == "optimized" and stage == "simulate":
                module, cwd = "optimized", ROOT
            args = [sys.executable, "-W", "always", "-m", module, "--output", temporary]
            if stage == "simulate":
                args += ["--config", str(config_path)]
                if engine == "baseline":
                    args += ["--force"]
                if engine == "optimized":
                    args += ["--workers", str(workers)]
            start = time.perf_counter()
            completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
            timings[stage] = time.perf_counter() - start
            if completed.returncode:
                raise RuntimeError(f"{engine} {stage} failed:\n{completed.stderr}")
            timings[f"{stage}_warnings"] = [
                line.strip()
                for line in completed.stderr.splitlines()
                if "Warning:" in line or "RuntimeWarning" in line
            ]
    timings["total"] = sum(timings[stage] for stage in ("simulate", "analyze", "figures"))
    return timings


def profile(engine, config, output):
    """Save cumulative and self-time profiles for an in-memory full study run."""
    profiler = cProfile.Profile()
    profiler.runcall(timed_study, engine, config)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(30)
    stats.sort_stats("tottime").print_stats(20)
    (output / f"profile_{engine}.txt").write_text(stream.getvalue().rstrip() + "\n")


def fit_complexity(measurements, output):
    """Fit log-log slopes using the three large sizes (m >= 2000).

    Keep the full-range slopes alongside the fitted range so different
    measurement runs and fitting choices can be distinguished.
    """
    import numpy as np

    fitted = {}
    for engine, group in measurements.groupby("engine"):
        medians = group.groupby("m").seconds.median()
        tail = medians.loc[medians.index >= 2000]
        if len(tail) < 3:
            raise ValueError("Complexity fitting needs at least three sizes at m >= 2000")
        slope, intercept = np.polyfit(np.log(tail.index), np.log(tail), 1)
        fitted[engine] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "sizes": tail.index.tolist(),
            "full_range_slope": float(
                np.polyfit(np.log(medians.index), np.log(medians), 1)[0]
            ),
        }
    report = {"fit_min_m": 2000, "engines": fitted}
    (output / "complexity_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "complexity_slopes.json").write_text(
        json.dumps({engine: result["slope"] for engine, result in fitted.items()}, indent=2)
        + "\n"
    )
    return report


def complexity(engines, config, trials, output):
    """Measure the same two scenarios per size and fit a log-log slope."""
    rows = []
    reduced = dict(config, pi0_values=[0.2, 0.8], boundary_pi0=[], replicates=100)
    for size in SIZES:
        for trial in range(trials):
            for engine in engines[:: (-1 if trial % 2 else 1)]:
                elapsed, _ = timed_study(engine, dict(reduced, m=size))
                rows.append(
                    {"engine": engine, "m": size, "trial": trial + 1, "seconds": elapsed}
                )
    measurements = pd.DataFrame(rows)
    measurements.to_csv(output / "complexity.csv", index=False)
    fit_complexity(measurements, output)


def stability(engines, config, output):
    """Capture numerical warnings and exceptions under study and stress conditions."""
    records = []
    for name, settings in (
        ("study", config),
        (
            "small_tail",
            dict(config, m=2000, replicates=30, gammas=[1e-12, 1e-8], effect=12.0),
        ),
    ):
        for engine in engines:
            result = None
            error = None
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                try:
                    _, result = timed_study(engine, settings)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            counts = {}
            for warning in observed:
                key = f"{warning.category.__name__}: {warning.message}"
                counts[key] = counts.get(key, 0) + 1
            records.append(
                {
                    "engine": engine,
                    "case": name,
                    "warnings": counts,
                    "error": error,
                    "rows": None if result is None else len(result),
                    "undefined_power_rows": None
                    if result is None
                    else int(result.power.isna().sum()),
                }
            )
    (output / "stability.json").write_text(json.dumps(records, indent=2) + "\n")
    if any(record["error"] for record in records):
        raise RuntimeError("A stability run failed; see stability.json")


def benchmark(engines, config_path, trials, output):
    """Alternate baseline and optimized full runs after prewarming shared imports."""
    # Warm filesystem/font caches without recording a measured run.
    with tempfile.TemporaryDirectory(prefix="storey-warmup-") as temporary:
        config = dict(load_config(config_path), replicates=10)
        warmup = Path(temporary) / "warmup.json"
        warmup.write_text(json.dumps(config))
        run_full("baseline", warmup, 1)
    runs = []
    for trial in range(trials):
        for engine in engines[:: (-1 if trial % 2 else 1)]:
            result = run_full(engine, config_path, 1)
            runs.append({"engine": engine, "trial": trial + 1, **result})
    medians = {
        engine: {
            phase: statistics.median(run[phase] for run in runs if run["engine"] == engine)
            for phase in ("simulate", "analyze", "figures", "total")
        }
        for engine in engines
    }
    report = {"runs": runs, "medians": medians}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


def phase_timings(engines, config, trials, output):
    """Separate in-memory computation from CSV formatting, gzip and disk writing."""
    importlib.import_module("optimized")
    warmup = dict(config, replicates=2)
    for engine in engines:
        timed_study(engine, warmup)
    rows = []
    with tempfile.TemporaryDirectory(prefix="storey-phase-") as temporary:
        archive = Path(temporary) / "replicates.csv.gz"
        for trial in range(trials):
            for engine in engines[:: (-1 if trial % 2 else 1)]:
                elapsed, result = timed_study(engine, config)
                start = time.perf_counter()
                result.to_csv(
                    archive,
                    index=False,
                    float_format="%.17g",
                    compression={"method": "gzip", "mtime": 0},
                )
                rows.append(
                    {
                        "engine": engine,
                        "trial": trial + 1,
                        "compute": elapsed,
                        "csv_gzip_write": time.perf_counter() - start,
                        "rows": len(result),
                    }
                )
    medians = {
        engine: {
            phase: statistics.median(row[phase] for row in rows if row["engine"] == engine)
            for phase in ("compute", "csv_gzip_write")
        }
        for engine in engines
    }
    (output / "phases.json").write_text(
        json.dumps({"runs": rows, "medians": medians}, indent=2) + "\n"
    )


def parallel_scaling(config, trials, output):
    """Time the optimized in-memory study including pool startup and output transfer."""
    rows = []
    for trial in range(trials):
        for workers in WORKERS[:: (-1 if trial % 2 else 1)]:
            elapsed, _ = timed_study("optimized", config, workers)
            rows.append({"workers": workers, "trial": trial + 1, "seconds": elapsed})
    pd.DataFrame(rows).to_csv(output / "parallel.csv", index=False)


def main():
    """Dispatch one experiment and save environment details with the measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=(
            "profile",
            "complexity",
            "benchmark",
            "stability",
            "scaling",
            "baseline",
            "phases",
        ),
    )
    parser.add_argument(
        "--engine", choices=("baseline", "optimized", "both"), default="both"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/study.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()
    config = load_config(config_path)
    engines = ("baseline", "optimized") if args.engine == "both" else (args.engine,)
    if args.task in ("profile", "baseline"):
        for engine in ("baseline",) if args.task == "baseline" else engines:
            profile(engine, config, output)
    if args.task in ("complexity", "baseline"):
        complexity(
            ("baseline",) if args.task == "baseline" else engines,
            config,
            args.trials,
            output,
        )
    if args.task in ("benchmark", "baseline"):
        benchmark(
            ("baseline",) if args.task == "baseline" else engines,
            config_path,
            args.trials,
            output,
        )
    if args.task in ("stability", "baseline"):
        stability(("baseline",) if args.task == "baseline" else engines, config, output)
    if args.task == "phases":
        phase_timings(engines, config, args.trials, output)
    if args.task == "scaling":
        parallel_scaling(config, args.trials, output)
    environment = {
        "baseline_commit": BASELINE_COMMIT,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "pandas", "matplotlib")
        },
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


if __name__ == "__main__":
    main()
