import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from its_nn_value_first_grouped.config import add_args as add_clean4_args

from .baseline_rf import RandomForestPolicy
from .baseline_svm import SVMPolicy
from .baseline_linear import LinearPolicy
from .baseline_initial_nn import InitialNetworkPolicy
from .clean4_policy import ValueFirstPolicy
from .data import generate_synthetic_data_with_truth
from .eval_policy import evaluate_policy
from .plot_compare import plot_nn_advantage, plot_summary_panels


def parse_groups(groups_str: str, n_actions: int):
    groups = []
    for block in groups_str.strip().split("|"):
        g = [int(z.strip()) for z in block.split(",") if z.strip() != ""]
        groups.append(g)
    flat = sorted(a for g in groups for a in g)
    if flat != list(range(n_actions)):
        raise ValueError(f"Groups do not cover 0..{n_actions-1}: {groups}")
    return groups


def group_ids_from_groups(groups, n_actions):
    out = np.full(int(n_actions), -1, dtype=int)
    for gid, grp in enumerate(groups):
        for a in grp:
            out[int(a)] = gid
    return out


def generate_split(args, scenario, rep_seed):
    groups = parse_groups(args.groups, args.n_actions)
    common = dict(
        n_features=args.n_features,
        n_treatments=args.n_actions,
        groups=groups,
        scenario=scenario,
        sigma=args.y_noise,
        treat_random=(scenario != "observational"),
        baseline_scale=args.baseline_scale,
        effect_scale=args.effect_scale,
        near_fused_eps=args.near_fused_eps,
        near_fused_seed=args.near_fused_seed + rep_seed,
        group_effect_style=args.group_effect_style,
        baseline_style=args.baseline_style,
        treatment_prob_style=args.treatment_prob_style,
        group_prototypes=args.group_prototypes,
        treatment_probs=args.treatment_probs,
        ensure_all_treatments=bool(getattr(args, "ensure_all_treatments", False)),
    )
    train = generate_synthetic_data_with_truth(
        n_samples=args.n_train, seed=args.seed_train + rep_seed, **common
    )
    val = generate_synthetic_data_with_truth(
        n_samples=args.n_val, seed=args.seed_val + rep_seed, **common
    )
    test = generate_synthetic_data_with_truth(
        n_samples=args.n_test, seed=args.seed_test + rep_seed, **common
    )
    return train, val, test, groups


def build_methods(args, rep_seed, groups, outdir):
    true_group_ids = group_ids_from_groups(groups, args.n_actions)
    available = {
        "value_first_nn": ValueFirstPolicy(
            copy.deepcopy(args),
            true_group_ids=true_group_ids,
            outdir=outdir,
        ),
        "initial_nn": InitialNetworkPolicy(
            copy.deepcopy(args),
        ),
        "ridge": LinearPolicy(
            n_actions=args.n_actions,
            model_type="ridge",
            random_state=rep_seed,
            refit_on_trainval=True,
        ),
        "lasso": LinearPolicy(
            n_actions=args.n_actions,
            model_type="lasso",
            random_state=rep_seed,
            refit_on_trainval=True,
        ),
        "elasticnet": LinearPolicy(
            n_actions=args.n_actions,
            model_type="elasticnet",
            random_state=rep_seed,
            refit_on_trainval=True,
        ),
        "rf": RandomForestPolicy(
            n_actions=args.n_actions,
            random_state=rep_seed,
            n_jobs=args.rf_n_jobs,
            refit_on_trainval=True,
        ),
        "svm": SVMPolicy(
            n_actions=args.n_actions,
            refit_on_trainval=True,
        ),
    }
    selected = [name.strip() for name in str(args.methods).split(",") if name.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")
    return {name: available[name] for name in selected}


def run_one_rep(args, scenario, rep_seed, outdir):
    train, val, test, groups = generate_split(args, scenario, rep_seed)
    Xtr = train["X"].cpu().numpy().astype(np.float32)
    Atr = train["A"].cpu().numpy().astype(np.int64)
    Ytr = train["y"].cpu().numpy().astype(np.float32)
    Xval = val["X"].cpu().numpy().astype(np.float32)
    Aval = val["A"].cpu().numpy().astype(np.int64)
    Yval = val["y"].cpu().numpy().astype(np.float32)
    Xte = test["X"].cpu().numpy().astype(np.float32)
    Ate = test["A"].cpu().numpy().astype(np.int64)
    Yte = test["y"].cpu().numpy().astype(np.float32)
    mu_te = test["mu_true"].cpu().numpy().astype(np.float32)
    aopt_te = test["a_opt"].cpu().numpy().astype(np.int64)

    rows = []
    rep_outdir = outdir / f"{scenario}_value_first_nn_seed{rep_seed}"
    for method_name, method in build_methods(args, rep_seed, groups, rep_outdir).items():
        print(f"[Run] scenario={scenario} seed={rep_seed} method={method_name}", flush=True)
        method.fit(Xtr, Atr, Ytr, X_val=Xval, A_val=Aval, Y_val=Yval)
        a_hat = np.asarray(method.predict_treatment(Xte)).astype(np.int64)
        mu_hat = np.asarray(method.predict_mu(Xte)).astype(np.float64)
        factual_pred = mu_hat[np.arange(len(Ate)), Ate]
        factual_test_mse = float(np.mean((Yte - factual_pred) ** 2))
        metrics = evaluate_policy(mu_te, a_hat, aopt_te)
        diag = method.get_diagnostics() if hasattr(method, "get_diagnostics") else {}
        row = {
            "scenario": scenario,
            "seed": int(rep_seed),
            "method": method_name,
            "n_train": int(args.n_train),
            "n_val": int(args.n_val),
            "n_test": int(args.n_test),
            "n_features": int(args.n_features),
            "n_actions": int(args.n_actions),
            "y_noise": float(args.y_noise),
            "baseline_scale": float(args.baseline_scale),
            "effect_scale": float(args.effect_scale),
            "near_fused_eps": float(args.near_fused_eps),
            "near_fused_seed": int(args.near_fused_seed + rep_seed),
        }
        row.update(metrics)
        row["test_factual_mse"] = factual_test_mse
        row.update(diag)
        rows.append(row)
    return rows


def _to_float_or_nan(v):
    if v is None:
        return math.nan
    try:
        return float(v)
    except Exception:
        return math.nan


def _mean(xs):
    vals = [x for x in xs if not math.isnan(x)]
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def _std(xs):
    vals = [x for x in xs if not math.isnan(x)]
    if len(vals) <= 1:
        return 0.0 if vals else math.nan
    m = sum(vals) / len(vals)
    return float((sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5)


def summarize_results(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["method"])].append(row)

    summary_rows = []
    metric_cols = [
        "value",
        "oracle_value",
        "regret",
        "acc_opt",
        "test_factual_mse",
        "value_first_final_val_mse",
        "value_first_final_test_proxy_mse",
        "value_first_final_n_groups",
        "value_first_exact_recovery",
        "value_first_pairwise_accuracy",
        "value_first_ari",
        "initial_nn_best_val_mse",
        "ridge_best_val_mse",
        "lasso_best_val_mse",
        "elasticnet_best_val_mse",
        "rf_best_val_mse",
        "svm_best_val_mse",
    ]

    for (scenario, method), rs in sorted(grouped.items()):
        out = {
            "scenario": scenario,
            "method": method,
            "n_rep": len(rs),
        }
        for col in metric_cols:
            vals = [_to_float_or_nan(r.get(col)) for r in rs]
            if any(not math.isnan(v) for v in vals):
                out[f"{col}_mean"] = _mean(vals)
                out[f"{col}_sd"] = _std(vals)
        summary_rows.append(out)

    summary_rows.sort(key=lambda r: (r["scenario"], -(r.get("value_mean", -math.inf))))
    return summary_rows


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
    widths = {}
    for c in cols:
        widths[c] = max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows))
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    lines = [header]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if math.isnan(val) else val
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return None if math.isnan(x) else x
    return x


def get_args():
    parser = argparse.ArgumentParser(description="Formal comparison: Value-first NN vs RF vs SVM")
    parser = add_clean4_args(parser)
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="linear_easy,nonlinear_easy,nonlinear_hard",
    )
    parser.add_argument("--groups", type=str, default="0,1,2,3,4|5,6,7,8,9")
    parser.add_argument(
        "--methods",
        type=str,
        default="value_first_nn,initial_nn,ridge,lasso,elasticnet,rf,svm",
        help="Comma-separated subset of methods to run.",
    )
    parser.add_argument("--rf_n_jobs", type=int, default=-1)
    return parser.parse_args()


def main():
    args = get_args()
    outdir = Path(args.out_base) / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "compare_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    rep_seeds = [args.seed0 + i for i in range(args.n_reps)]
    all_rows = []

    for scenario in scenarios:
        for rep_seed in rep_seeds:
            all_rows.extend(run_one_rep(args, scenario, rep_seed, outdir))
            _write_csv(outdir / "raw_results.csv", all_rows)

    summary_rows = summarize_results(all_rows)
    _write_csv(outdir / "raw_results.csv", all_rows)
    _write_csv(outdir / "summary_results.csv", summary_rows)
    plot_summary_panels(summary_rows, outdir)
    plot_nn_advantage(all_rows, outdir)
    with open(outdir / "summary.json", "w") as f:
        json.dump(
            _to_jsonable(
                {
                    "config": vars(args),
                    "raw_rows": all_rows,
                    "summary_rows": summary_rows,
                }
            ),
            f,
            indent=2,
        )
    with open(outdir / "summary_results.txt", "w") as f:
        f.write(_format_table(summary_rows))
        f.write("\n")
    print(_format_table(summary_rows), flush=True)


if __name__ == "__main__":
    main()
