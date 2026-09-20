import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _f(row, key, default=np.nan):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def plot_summary_panels(summary_rows, outdir: Path):
    scenarios = sorted({r["scenario"] for r in summary_rows})
    methods = ["value_first_nn", "initial_nn", "ridge", "lasso", "elasticnet", "rf", "svm"]
    metric_specs = [
        ("value_mean", "Value", True),
        ("regret_mean", "Regret", False),
        ("acc_opt_mean", "Acc. of Optimal Action", True),
        ("test_factual_mse_mean", "Test Factual MSE", False),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(scenarios), figsize=(4.8 * len(scenarios), 3.8 * len(metric_specs)))
    if len(metric_specs) == 1 and len(scenarios) == 1:
        axes = np.array([[axes]])
    elif len(metric_specs) == 1:
        axes = np.array([axes])
    elif len(scenarios) == 1:
        axes = np.array([[ax] for ax in axes])

    colors = {
        "value_first_nn": "#1f77b4",
        "initial_nn": "#9467bd",
        "ridge": "#8c564b",
        "lasso": "#e377c2",
        "elasticnet": "#7f7f7f",
        "rf": "#2ca02c",
        "svm": "#ff7f0e",
    }

    for c, scenario in enumerate(scenarios):
        block = {r["method"]: r for r in summary_rows if r["scenario"] == scenario}
        for r, (metric, title, higher_better) in enumerate(metric_specs):
            ax = axes[r, c]
            vals = [_f(block.get(m, {}), metric, np.nan) for m in methods]
            sds = [_f(block.get(m, {}), metric.replace("_mean", "_sd"), 0.0) for m in methods]
            labels = ["NN", "InitNN", "Ridge", "Lasso", "ENet", "RF", "SVM"]
            ax.bar(labels, vals, yerr=sds, color=[colors[m] for m in methods], alpha=0.9, capsize=4)
            ax.set_title(f"{scenario} | {title}")
            ax.grid(axis="y", alpha=0.25)
            if not higher_better:
                best = np.nanmin(vals)
            else:
                best = np.nanmax(vals)
            for i, v in enumerate(vals):
                if np.isfinite(v) and abs(v - best) < 1e-12:
                    ax.text(i, v, " best", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "scenario_method_panels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nn_advantage(raw_rows, outdir: Path):
    by_scenario_method = defaultdict(list)
    for r in raw_rows:
        by_scenario_method[(r["scenario"], r["method"])].append(r)

    scenarios = sorted({r["scenario"] for r in raw_rows})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 4.2))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        nn_rows = sorted(by_scenario_method.get((scenario, "value_first_nn"), []), key=lambda r: int(r["seed"]))
        rf_rows = sorted(by_scenario_method.get((scenario, "rf"), []), key=lambda r: int(r["seed"]))
        svm_rows = sorted(by_scenario_method.get((scenario, "svm"), []), key=lambda r: int(r["seed"]))
        n = min(len(nn_rows), len(rf_rows), len(svm_rows))
        x = np.arange(n)
        nn_vs_rf = [_f(nn_rows[i], "value") - _f(rf_rows[i], "value") for i in range(n)]
        nn_vs_svm = [_f(nn_rows[i], "value") - _f(svm_rows[i], "value") for i in range(n)]
        ax.axhline(0.0, color="black", lw=1, alpha=0.6)
        ax.plot(x, nn_vs_rf, marker="o", label="NN - RF")
        ax.plot(x, nn_vs_svm, marker="s", label="NN - SVM")
        ax.set_title(f"{scenario} | Value Advantage")
        ax.set_xlabel("Replication")
        ax.set_ylabel("Higher is better for NN")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "nn_advantage_by_scenario.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_boxplots(raw_rows, outdir: Path):
    scenarios = sorted({r["scenario"] for r in raw_rows})
    methods = ["value_first_nn", "initial_nn", "ridge", "lasso", "elasticnet", "rf", "svm"]
    labels = ["NN", "InitNN", "Ridge", "Lasso", "ENet", "RF", "SVM"]
    metric_specs = [
        ("value", "Value", True),
        ("regret", "Regret", False),
        ("acc_opt", "Acc. of Optimal Action", True),
        ("test_factual_mse", "Test Factual MSE", False),
    ]
    colors = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#2ca02c", "#ff7f0e"]

    fig, axes = plt.subplots(len(metric_specs), len(scenarios), figsize=(4.8 * len(scenarios), 3.8 * len(metric_specs)))
    if len(metric_specs) == 1 and len(scenarios) == 1:
        axes = np.array([[axes]])
    elif len(metric_specs) == 1:
        axes = np.array([axes])
    elif len(scenarios) == 1:
        axes = np.array([[ax] for ax in axes])

    for c, scenario in enumerate(scenarios):
        block = [r for r in raw_rows if r["scenario"] == scenario]
        for r, (metric, title, higher_better) in enumerate(metric_specs):
            ax = axes[r, c]
            data = [[_f(row, metric) for row in block if row["method"] == m] for m in methods]
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_title(f"{scenario} | {title}")
            ax.grid(axis="y", alpha=0.25)
            if not higher_better:
                best = min(np.nanmedian(d) for d in data)
            else:
                best = max(np.nanmedian(d) for d in data)
            ax.axhline(best, color="black", lw=0.8, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(outdir / "scenario_method_boxplots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_curves(run_dir: Path):
    seed_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and "value_first_nn_seed" in p.name])
    for seed_dir in seed_dirs:
        csv_files = sorted(seed_dir.glob("refit_metrics_cycle*.csv"))
        if not csv_files:
            continue
        n = len(csv_files)
        fig, axes = plt.subplots(n, 1, figsize=(8.0, 3.4 * n), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, csv_path in zip(axes, csv_files):
            rows = _read_csv(csv_path)
            lambdas = [_f(r, "lambda") for r in rows]
            mses = [_f(r, "refit_val_mse") for r in rows]
            groups = [_f(r, "n_groups") for r in rows]
            finite = [m for m in mses if np.isfinite(m)]
            best = np.nanmin(mses) if finite else np.nan
            ax.plot(lambdas, mses, marker="o", color="#1f77b4", label="Refit val MSE")
            best_idx = int(np.nanargmin(mses)) if finite else None
            if best_idx is not None:
                ax.scatter([lambdas[best_idx]], [mses[best_idx]], color="#d62728", s=45, zorder=3, label="Best")
                ax.annotate(
                    f"best: g={int(groups[best_idx])}",
                    (lambdas[best_idx], mses[best_idx]),
                    textcoords="offset points",
                    xytext=(8, 6),
                    fontsize=9,
                )
            ax.set_xscale("log")
            ax.set_ylabel("Val MSE")
            ax.set_title(csv_path.stem.replace("_", " "))
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
        axes[-1].set_xlabel("Lambda (log scale)")
        fig.tight_layout()
        fig.savefig(seed_dir / "cycle_lambda_mse_panels.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot comparison outputs for value-first NN vs RF/SVM.")
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_rows = _read_csv(run_dir / "raw_results.csv")
    summary_rows = _read_csv(run_dir / "summary_results.csv")
    plot_summary_panels(summary_rows, run_dir)
    plot_nn_advantage(raw_rows, run_dir)
    plot_metric_boxplots(raw_rows, run_dir)
    plot_lambda_curves(run_dir)


if __name__ == "__main__":
    main()
