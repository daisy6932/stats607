"""Static figures with Monte Carlo uncertainty and explicit comparison labels."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"storey_fixed": "#007A87", "bh_estimated": "#BD4D24", "bh_fixed": "#6750A4"}
LABELS = {
    "storey_fixed": "Storey: fixed region",
    "bh_estimated": "BH: same-data estimated level",
    "bh_fixed": "BH: prespecified 0.05",
}


def configure_plot_style():
    """Set the shared Matplotlib defaults for the result figures."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
        }
    )


def plot_with_mc_intervals(axis, data, label, color, **kwargs):
    """Draw a labeled mean and 95% Monte Carlo interval series on axis.

    Data contains pi0, mean and mcse columns. Extra options go to errorbar."""
    data = data.sort_values("pi0")
    axis.errorbar(
        data.pi0,
        data["mean"],
        yerr=1.96 * data.mcse,
        label=label,
        color=color,
        marker="o",
        markersize=4,
        capsize=2,
        linewidth=1.5,
        **kwargs,
    )


def save_figure(figure, figure_dir, name, footer):
    """Add footer text, write figure to PNG and SVG under figure_dir, then close it."""
    figure.text(0.02, 0.012, footer, fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.075, 1, 0.95))
    for extension in ("png", "svg"):
        figure.savefig(figure_dir / f"{name}.{extension}", bbox_inches="tight")
    plt.close(figure)


def make_figures(results_dir, reference_path):
    """Read summary tables from results_dir and Table 2 from reference_path.

    Write the four study figures to the figures subdirectory."""
    configure_plot_style()
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results_dir / "summary.csv")
    diagnostics = pd.read_csv(results_dir / "diagnostics.csv")
    paper_table = pd.read_csv(reference_path)
    gammas = sorted(summary.gamma.unique(), reverse=True)
    num_replicates = int(pd.read_csv(results_dir / "design_matrix.csv").replicates.iloc[0])
    figure, axes = plt.subplots(1, len(gammas), figsize=(11, 4.6), squeeze=False)
    for axis, gamma in zip(axes[0], gammas):
        subset = summary[
            (summary.role == "primary")
            & (summary.gamma == gamma)
            & (summary.metric == "power")
        ]
        for method in ("storey_fixed", "bh_estimated"):
            plot_with_mc_intervals(
                axis, subset[subset.method == method], LABELS[method], COLORS[method]
            )
        paper_points = paper_table[paper_table.gamma == gamma]
        axis.scatter(
            paper_points.pi0,
            paper_points.paper_storey_power,
            facecolors="none",
            edgecolors=COLORS["storey_fixed"],
            marker="s",
            s=40,
            label="Paper: Storey",
        )
        axis.scatter(
            paper_points.pi0,
            paper_points.paper_bh_power,
            color=COLORS["bh_estimated"],
            marker="x",
            s=35,
            label="Paper: BH",
        )
        axis.set(
            title=f"Fixed rejection threshold γ = {gamma:g}",
            xlabel="True null proportion (π₀)",
            ylabel="Mean power",
            ylim=(0, 0.43),
        )
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Reproducing Storey (2002), Table 2: power", fontsize=14)
    save_figure(
        figure,
        figure_dir,
        "power_comparison",
        f"Independent N(0,1)/N(2,1) tests; m=1000; λ=0.5; {num_replicates} replicates/condition.\n"
        "Bars: pointwise 95% Monte Carlo intervals. BH uses Storey's FDR estimate.",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, estimate, title in zip(
        axes,
        ("pi0_hat", "fdr_hat"),
        ("Null proportion estimation bias", "Fixed-region FDR estimation bias"),
    ):
        for gamma, color in zip(gammas, ("#007A87", "#BD4D24")):
            data = diagnostics[
                (diagnostics.role == "primary")
                & (diagnostics.gamma == gamma)
                & (diagnostics.estimate == estimate)
            ]
            plot_with_mc_intervals(axis, data, f"γ = {gamma:g}", color)
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set(
            title=title,
            xlabel="True null proportion (π₀)",
            ylabel="Mean estimate minus known truth",
        )
        axis.legend(fontsize=8)
    figure.suptitle("Diagnostics: estimator bias with Monte Carlo uncertainty", fontsize=14)
    save_figure(
        figure,
        figure_dir,
        "estimator_diagnostics",
        f"Fixed-count design; {num_replicates} replicates/condition. Bars: pointwise 95% Monte Carlo intervals.\n"
        "FDR truth uses an exact binomial integral. The π₀ curves share λ and datasets.",
    )

    figure, axes = plt.subplots(1, len(gammas), figsize=(11, 4.6), squeeze=False)
    for axis, gamma in zip(axes[0], gammas):
        subset = summary[
            (summary.role == "primary")
            & (summary.gamma == gamma)
            & (summary.metric == "fdp")
        ]
        for method in COLORS:
            plot_with_mc_intervals(
                axis, subset[subset.method == method], LABELS[method], COLORS[method]
            )
        axis.axhline(
            0.05, color="gray", linewidth=1, linestyle="--", label="0.05 reference"
        )
        axis.set(
            title=f"γ = {gamma:g}",
            xlabel="True null proportion (π₀)",
            ylabel="Empirical FDR (mean FDP)",
        )
        axis.legend(fontsize=8)
    figure.suptitle(
        "Error rates: the paper comparison does not match realized FDR", fontsize=14
    )
    save_figure(
        figure,
        figure_dir,
        "empirical_fdr",
        f"Independent tests; {num_replicates} replicates/condition. Bars: pointwise 95% Monte Carlo intervals.\n"
        "The purple series shows BH at a prespecified level of 0.05.",
    )

    figure, axes = plt.subplots(1, len(gammas), figsize=(11, 4.6), squeeze=False)
    for axis, gamma in zip(axes[0], gammas):
        for mode, linestyle in (("fixed", "-"), ("iid", "--")):
            for method in ("storey_fixed", "bh_estimated"):
                subset = summary[
                    (summary["mode"] == mode)
                    & (summary.gamma == gamma)
                    & (summary.metric == "power")
                    & (summary.role != "boundary")
                    & (summary.method == method)
                ]
                plot_with_mc_intervals(
                    axis,
                    subset,
                    f"{LABELS[method]} ({mode})",
                    COLORS[method],
                    linestyle=linestyle,
                )
        axis.set(
            title=f"γ = {gamma:g}",
            xlabel="True null proportion (π₀)",
            ylabel="Mean power",
            ylim=(0, 0.43),
        )
        axis.legend(fontsize=7)
    figure.suptitle("Sensitivity to fixed versus random hypothesis labels", fontsize=14)
    save_figure(
        figure,
        figure_dir,
        "label_sensitivity",
        f"Fixed counts and iid labels; {num_replicates} replicates/condition.\n"
        "Bars: pointwise 95% Monte Carlo intervals.",
    )


def main():
    """Parse output and reference paths and generate the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--reference", default="results/reference/table2.csv")
    args = parser.parse_args()
    make_figures(Path(args.output), args.reference)
    print(f"Saved four PNG/SVG figures in {args.output}/figures")


if __name__ == "__main__":
    main()
