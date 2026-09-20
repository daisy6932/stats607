"""Reproduce a small NSCF simulation using the original research implementation."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def validate_results(frame, cfg):
    """Reject incomplete cells, nonfinite metrics, or inconsistent policy values."""
    expected = {(s, m) for s in range(cfg["seed0"], cfg["seed0"] + cfg["n_reps"])
                for m in cfg["methods"].split(",")}
    required = {"seed", "method", "regret", "value", "oracle_value", "acc_opt"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing result columns: {required - set(frame.columns)}")
    cells = list(zip(frame.seed, frame.method))
    if len(cells) != len(expected) or set(cells) != expected:
        raise ValueError("Missing or duplicate seed-by-method cells")
    metrics = frame[["regret", "value", "oracle_value", "acc_opt"]]
    if not np.isfinite(metrics.to_numpy()).all():
        raise ValueError("Nonfinite policy metrics")
    if (frame.regret < -1e-6).any() or not frame.acc_opt.between(0, 1).all():
        raise ValueError("Regret or optimal-action accuracy outside valid range")
    if not np.allclose(frame.regret, frame.oracle_value - frame.value, atol=1e-6):
        raise ValueError("Regret must equal oracle value minus policy value")


def reproduce(output_root=ROOT):
    """Run the simulation from seeds, validate its outputs, and report results."""
    root = Path(output_root).resolve()
    cfg = json.loads((ROOT / "config/experiment.json").read_text())
    for folder in ["data/processed", "artifacts", "results/tables", "results/figures", "results/report"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [sys.executable, "-u", "-m", "its_compare_value_first_2_grouped.sim_compare_formal",
               "--out_base", str(root / "artifacts"), "--run_name", "simulation"]
    for key, value in cfg.items():
        command += [f"--{key}"] + ([] if value is True else [str(value)])
    print("[1/3] Training NSCF, Neural Q, and ridge on three fixed seeds (CPU)", flush=True)
    print(f"Detailed progress: {root / 'artifacts/training.log'}", flush=True)
    with (root / "artifacts/training.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = (root / "artifacts/training.log").read_text()[-5000:]
        raise RuntimeError(f"Simulation failed (exit {result.returncode}). Log tail:\n{tail}")
    print("[2/3] Validating seed-level metrics and aggregating", flush=True)
    frame = pd.read_csv(root / "artifacts/simulation/raw_results.csv")
    validate_results(frame, cfg)
    columns = ["scenario", "seed", "method", "value", "oracle_value", "regret", "acc_opt"]
    frame[columns].to_csv(root / "data/processed/policy_metrics.csv", index=False)
    summary = frame.groupby("method", sort=True).agg(
        n=("regret", "size"), mean_regret=("regret", "mean"), sd_regret=("regret", "std"),
        mean_value=("value", "mean"), mean_accuracy=("acc_opt", "mean")).reset_index()
    summary.to_csv(root / "results/tables/policy_comparison.csv", index=False)
    print("[3/3] Writing final figure and report", flush=True)
    labels = {"value_first_nn": "NSCF", "initial_nn": "Neural Q", "ridge": "Ridge"}
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for position, method in enumerate(summary.method):
        vals = frame.loc[frame.method == method, "regret"].to_numpy()
        ax.scatter(np.full(len(vals), position), vals, color="#6484a6", s=40)
        ax.scatter(position, vals.mean(), color="#b33b32", marker="D", s=60, zorder=3)
    ax.set_xticks(range(len(summary)), [labels[m] for m in summary.method])
    ax.set_ylabel("Policy regret (lower is better)")
    ax.set_title("Small synthetic demonstration: 3 seeds\nDots: individual seeds; diamonds: means")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(root / "results/figures/policy_regret.png", dpi=160)
    plt.close(fig)
    (root / "results/report/summary.md").write_text(
        "# ITR Phase 1 reproducibility demonstration\n\n"
        f"Seeds: {cfg['seed0']} through {cfg['seed0'] + cfg['n_reps'] - 1}; "
        f"train/validation/test: {cfg['n_train']}/{cfg['n_val']}/{cfg['n_test']}.\n\n"
        "This small optimization budget demonstrates reproducibility of the original "
        "NSCF implementation. Three repetitions do not support general method rankings, "
        "and these are not the paper-scale simulation or PDX results. "
        "The table reports between-seed sample standard deviations, not confidence intervals.\n")
    print(f"Completed: {root / 'results'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    reproduce(parser.parse_args().output_root)
