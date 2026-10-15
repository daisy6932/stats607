"""Static figures with Monte Carlo uncertainty and explicit comparison labels."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"storey_fixed": "#007A87", "bh_estimated": "#BD4D24", "bh_fixed": "#6750A4"}
LABELS = {"storey_fixed": "Storey: fixed region", "bh_estimated": "BH: same-data estimated level",
          "bh_fixed": "BH: prespecified 0.05"}


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 140,
                         "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def draw(ax, data, label, color, **kwargs):
    data = data.sort_values("pi0")
    ax.errorbar(data.pi0, data["mean"], yerr=1.96 * data.mcse, label=label,
                color=color, marker="o", markersize=4, capsize=2, linewidth=1.5, **kwargs)


def save(fig, out, name, footer):
    fig.text(0.02, 0.012, footer, fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))
    for extension in ("png", "svg"):
        fig.savefig(out / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def make_figures(root, reference):
    style()
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "summary.csv")
    diag = pd.read_csv(root / "diagnostics.csv")
    refs = pd.read_csv(reference)
    gammas = sorted(summary.gamma.unique(), reverse=True)
    reps = int(pd.read_csv(root / "design_matrix.csv").replicates.iloc[0])
    fig, axes = plt.subplots(1, len(gammas), figsize=(11, 4.6), squeeze=False)
    for ax, gamma in zip(axes[0], gammas):
        subset = summary[(summary.role == "primary") & (summary.gamma == gamma) & (summary.metric == "power")]
        for method in ("storey_fixed", "bh_estimated"):
            draw(ax, subset[subset.method == method], LABELS[method], COLORS[method])
        old = refs[refs.gamma == gamma]
        ax.scatter(old.pi0, old.paper_storey_power, facecolors="none", edgecolors=COLORS["storey_fixed"], marker="s", s=40, label="Paper: Storey")
        ax.scatter(old.pi0, old.paper_bh_power, color=COLORS["bh_estimated"], marker="x", s=35, label="Paper: BH")
        ax.set(title=f"Fixed rejection threshold gamma = {gamma:g}", xlabel="True null proportion (pi0)", ylabel="Mean power", ylim=(0, 0.43))
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Reproducing Storey (2002), Table 2: power", fontsize=14)
    save(fig, out, "power_comparison", f"Independent N(0,1)/N(2,1) tests; m=1000; lambda=0.5; {reps} replicates/condition.\nBars: pointwise 95% Monte Carlo intervals. BH uses Storey's estimate from the same data; realized FDR need not match.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, estimate, title in zip(axes, ("pi0_hat", "fdr_hat"), ("Null proportion estimation bias", "Fixed-region FDR estimation bias")):
        for gamma, color in zip(gammas, ("#007A87", "#BD4D24")):
            data = diag[(diag.role == "primary") & (diag.gamma == gamma) & (diag.estimate == estimate)]
            draw(ax, data, f"gamma={gamma:g}", color)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set(title=title, xlabel="True null proportion (pi0)", ylabel="Mean estimate minus known truth")
        ax.legend(fontsize=8)
    fig.suptitle("Diagnostics: estimator bias with Monte Carlo uncertainty", fontsize=14)
    save(fig, out, "estimator_diagnostics", f"Fixed-count design; {reps} replicates/condition. Bars: pointwise 95% Monte Carlo intervals.\nFDR truth uses an exact binomial integral. The pi0 curves coincide because lambda and datasets are shared.")

    fig, axes = plt.subplots(1, len(gammas), figsize=(11, 4.6), squeeze=False)
    for ax, gamma in zip(axes[0], gammas):
        subset = summary[(summary.role == "primary") & (summary.gamma == gamma) & (summary.metric == "fdp")]
        for method in COLORS:
            draw(ax, subset[subset.method == method], LABELS[method], COLORS[method])
        ax.axhline(0.05, color="gray", linewidth=1, linestyle="--", label="0.05 reference")
        ax.set(title=f"gamma = {gamma:g}", xlabel="True null proportion (pi0)", ylabel="Empirical FDR (mean FDP)")
        ax.legend(fontsize=8)
    fig.suptitle("Error rates: the paper comparison does not match realized FDR", fontsize=14)
    save(fig, out, "empirical_fdr", f"Independent tests; {reps} replicates/condition. Bars: pointwise 95% Monte Carlo intervals.\nOnly purple BH uses a prespecified 0.05 level. Fixed-region Storey has no common prespecified FDR level.")

    fig, axes = plt.subplots(1, len(gammas), figsize=(11, 4.6), squeeze=False)
    for ax, gamma in zip(axes[0], gammas):
        for mode, linestyle in (("fixed", "-"), ("iid", "--")):
            for method in ("storey_fixed", "bh_estimated"):
                subset = summary[(summary["mode"] == mode) & (summary.gamma == gamma) & (summary.metric == "power") & (summary.role != "boundary") & (summary.method == method)]
                draw(ax, subset, f"{LABELS[method]} ({mode})", COLORS[method], linestyle=linestyle)
        ax.set(title=f"gamma = {gamma:g}", xlabel="True null proportion (pi0)", ylabel="Mean power", ylim=(0, 0.43))
        ax.legend(fontsize=7)
    fig.suptitle("Sensitivity to fixed versus random hypothesis labels", fontsize=14)
    save(fig, out, "label_sensitivity", f"IID labels follow the errata; fixed counts follow our interpretation of Section 5. {reps} replicates/condition.\nBars: pointwise 95% Monte Carlo intervals; independent simulations across modes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--reference", default="results/reference/table2.csv")
    args = parser.parse_args()
    make_figures(Path(args.output), args.reference)
    print(f"Saved four PNG/SVG figures in {args.output}/figures")


if __name__ == "__main__":
    main()
