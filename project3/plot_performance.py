"""Plot end-to-end timings, complexity, and process scaling from measured files."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from performance import fit_complexity


def plot_performance(directory):
    """Read timing files from directory and write the comparison figure there."""
    benchmark = json.loads((directory / "benchmark.json").read_text())
    complexity = pd.read_csv(directory / "complexity.csv")
    fits = fit_complexity(complexity, directory)
    scaling = pd.read_csv(directory / "parallel.csv")
    colors = {"baseline": "#6A557A", "optimized": "#007A87"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    stages = ("simulate", "analyze", "figures", "total")
    positions = np.arange(len(stages))
    for offset, (engine, color) in zip((-0.18, 0.18), colors.items()):
        values = [benchmark["medians"][engine][stage] for stage in stages]
        bars = axes[0].bar(
            positions + offset, values, width=0.34, color=color, label=engine.capitalize()
        )
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[0].set_xticks(
        positions, ["Simulation", "Analysis", "Figures", "Total"], rotation=22
    )
    axes[0].set(title="Complete workflow", ylabel="Median wall time (s)")
    axes[0].set_ylim(0, 1.18 * axes[0].get_ylim()[1])
    axes[0].legend(frameon=False, fontsize=8)

    for engine, group in complexity.groupby("engine"):
        timing = group.groupby("m").seconds.agg(["median", "min", "max"])
        sizes, medians = timing.index.to_numpy(), timing["median"].to_numpy()
        fitted = fits["engines"][engine]
        slope, intercept = fitted["slope"], fitted["intercept"]
        fit_sizes = np.asarray(fitted["sizes"])
        color = colors[engine]
        axes[1].errorbar(
            sizes,
            medians,
            yerr=[medians - timing["min"], timing["max"] - medians],
            color=color,
            fmt="o",
            capsize=2,
            markersize=4,
            label=f"{engine.capitalize()} (slope {slope:.2f})",
        )
        axes[1].loglog(fit_sizes, np.exp(intercept) * fit_sizes**slope, "--", color=color)
    axes[1].set(
        title="Time as the number of tests grows",
        xlabel="Tests per dataset (m)",
        ylabel="Median in-memory time (s)",
    )
    size_ticks = sorted(complexity.m.unique())
    axes[1].set_xlim(0.82 * size_ticks[0], 1.22 * size_ticks[-1])
    axes[1].set_xticks(size_ticks, labels=[str(size) for size in size_ticks], fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)

    worker_times = scaling.groupby("workers").seconds.median()
    speedup = worker_times.loc[1] / worker_times
    axes[2].plot(speedup.index, speedup, "o-", color=colors["optimized"])
    axes[2].axhline(1, linestyle="--", color="#6b7280", linewidth=1)
    axes[2].set_xticks(speedup.index)
    axes[2].set(
        title="Process scaling", xlabel="Workers", ylabel="Speedup vs. optimized serial"
    )
    axes[2].set_ylim(0, max(1.2, 1.15 * speedup.max()))
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Storey simulation: baseline and optimized performance", fontsize=15)
    trials = int(complexity.trial.max())
    fig.text(
        0.02,
        0.02,
        f"Medians of {trials} runs. Middle: trial range; log-log fits use m ≥ 2000. "
        "Process timings include pool startup and result transfer.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(directory / "performance.png", dpi=200)
    plt.close(fig)


def main():
    """Accept the directory containing benchmark, complexity and scaling files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot_performance(parser.parse_args().directory)


if __name__ == "__main__":
    main()
