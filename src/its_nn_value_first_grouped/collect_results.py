import argparse
import csv
import json
from pathlib import Path


def _safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _flatten_summary(summary_path: Path):
    with summary_path.open("r") as f:
        summary = json.load(f)

    row = {
        "run_name": summary_path.parent.name,
        "summary_path": str(summary_path),
        "outdir": _safe_get(summary, "run", "outdir", default=""),
        "pretrain_val_mse": _safe_get(summary, "baseline", "pretrain_val_mse"),
        "anchor_val_mse": _safe_get(summary, "baseline", "anchor_val_mse"),
    }

    for block in [
        "prediction_best",
        "structure_best",
        "oracle_true_2group",
        "two_group_polish",
        "final_recommended",
    ]:
        prefix = block
        row[f"{prefix}_val_mse"] = _safe_get(summary, block, "val_mse")
        row[f"{prefix}_test_mse"] = _safe_get(summary, block, "test_mse")
        row[f"{prefix}_n_groups"] = _safe_get(summary, block, "n_groups")
        row[f"{prefix}_source"] = _safe_get(summary, block, "source")
        row[f"{prefix}_source_name"] = _safe_get(summary, block, "source_name")
        row[f"{prefix}_lambda_value"] = _safe_get(summary, block, "lambda_value")
        row[f"{prefix}_exact_recovery"] = _safe_get(summary, block, "grouping_metrics", "exact_recovery")
        row[f"{prefix}_pairwise_accuracy"] = _safe_get(summary, block, "grouping_metrics", "pairwise_accuracy")
        row[f"{prefix}_ari"] = _safe_get(summary, block, "grouping_metrics", "ari")

    truth_groups = _safe_get(summary, "truth", "true_groups", default=[])
    row["true_groups"] = json.dumps(truth_groups)
    row["final_recommended_selected_groups"] = json.dumps(
        _safe_get(summary, "final_recommended", "selected_groups", default=[])
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Directory containing run subfolders")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()

    root = Path(args.root)
    summary_paths = sorted(root.glob("*/summary.json"))
    rows = [_flatten_summary(path) for path in summary_paths]
    if not rows:
        raise FileNotFoundError(f"No summary.json files found under {root}")

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {args.out_csv} | n_runs={len(rows)}")


if __name__ == "__main__":
    main()
