import argparse
import csv
from pathlib import Path

from .plot_compare import (
    plot_metric_boxplots,
    plot_nn_advantage,
    plot_summary_panels,
)
from .sim_compare_formal import summarize_results


def _read_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_table(rows):
    rows = list(rows)
    if not rows:
        return ""
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    lines = [header]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def collect_seed_runs(root: Path, prefix: str):
    rows = []
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    for run_dir in run_dirs:
        raw_path = run_dir / "raw_results.csv"
        if raw_path.exists():
            rows.extend(_read_csv(raw_path))
    return rows, run_dirs


def main():
    parser = argparse.ArgumentParser(description="Aggregate many single-seed compare runs into one summary and boxplots.")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="value_first_h128_64_e600_seed")
    parser.add_argument("--out_name", type=str, default="value_first_h128_64_e600_reps50_aggregated")
    args = parser.parse_args()

    root = Path(args.root_dir)
    outdir = root / args.out_name
    outdir.mkdir(parents=True, exist_ok=True)

    raw_rows, run_dirs = collect_seed_runs(root, args.prefix)
    if not raw_rows:
        raise RuntimeError(f"No raw_results.csv found under {root} with prefix {args.prefix}")

    summary_rows = summarize_results(raw_rows)
    _write_csv(outdir / "raw_results.csv", raw_rows)
    _write_csv(outdir / "summary_results.csv", summary_rows)
    plot_summary_panels(summary_rows, outdir)
    plot_nn_advantage(raw_rows, outdir)
    plot_metric_boxplots(raw_rows, outdir)

    with open(outdir / "summary_results.txt", "w") as f:
        f.write(_format_table(summary_rows))
        f.write("\n")
    with open(outdir / "aggregation_manifest.txt", "w") as f:
        for p in run_dirs:
            f.write(str(p) + "\n")


if __name__ == "__main__":
    main()
