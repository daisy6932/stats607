import os
import json
import time
import random
import numpy as np
import torch

from .config import add_args
from .data import compute_true_mu_and_opt_actions, generate_synthetic_data
from .model import Net
from .solver_alt import run_alternating
from .plotting import (
    save_path_metrics_csv,
    plot_scaf_dendrogram,
    save_true_group_heatmap,
    save_partition_heatmap_from_group_ids,
    save_selected_partition_dendrogram_from_group_ids,
    plot_selected_cycle_path_with_group_bars,
    plot_cycle_summary,
    plot_refit_lambda_selection_curve,
    plot_refit_mse_vs_lambda,
    plot_groups_vs_lambda,
    plot_solution_path_dendrogram,
)
from .trainer import eval_mse
from .refit_selection import eval_grouped_model_on_original_actions


def make_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", buffering=1)

    def log_fn(msg):
        s = str(msg)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()

    return log_fn, f


def build_lambda_path(num_lambda, max_lambda):
    small = 1e-4
    num_points = int(num_lambda) - 1
    path = np.geomspace(small, float(max_lambda), num_points)
    return np.concatenate([np.array([0.0]), path])


def _group_ids_from_groups(groups, n_actions):
    gid = np.full(int(n_actions), -1, dtype=int)
    for j, grp in enumerate(groups):
        for a in grp:
            gid[int(a)] = int(j)
    if np.any(gid < 0):
        raise ValueError("Some treatments were not assigned to a true group.")
    return gid


def _oracle_diagnosis(prediction_best_val, oracle_true_val):
    pred = float(prediction_best_val)
    oracle = float(oracle_true_val)
    gap = oracle - pred
    near_abs = 0.5
    near_rel = 0.10 * max(pred, 1.0)
    if gap <= max(near_abs, near_rel):
        return (
            "Oracle true 2-group refit is close to prediction-best. "
            "This suggests the true grouping is competitive, and current search/"
            "selection may be leaving some grouping accuracy on the table."
        )
    if oracle < pred and gap <= max(1.0, 0.25 * max(pred, 1.0)):
        return (
            "Oracle true 2-group refit is somewhat worse than prediction-best, "
            "but still competitive. This suggests a real prediction-vs-grouping "
            "tradeoff rather than a simple coding bug."
        )
    return (
        "Oracle true 2-group refit is substantially worse than prediction-best. "
        "This suggests the current learned representation/objective prefers finer "
        "partitions, not just that the optimizer failed to find the true 2-group solution."
    )


def _group_ids_to_groups(group_ids):
    group_ids = np.asarray(group_ids).astype(int)
    groups = []
    for gid in np.unique(group_ids):
        members = np.where(group_ids == gid)[0].astype(int).tolist()
        groups.append(members)
    groups.sort(key=lambda grp: (len(grp), grp))
    return groups


def _comb2(n):
    n = int(n)
    return 0.5 * n * (n - 1)


def _pairwise_grouping_accuracy(pred_group_ids, true_group_ids):
    pred_group_ids = np.asarray(pred_group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)
    k = len(pred_group_ids)
    agree = 0
    total = 0
    for i in range(k):
        for j in range(i + 1, k):
            pred_same = pred_group_ids[i] == pred_group_ids[j]
            true_same = true_group_ids[i] == true_group_ids[j]
            if pred_same == true_same:
                agree += 1
            total += 1
    return float(agree / max(total, 1))


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
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, torch.Tensor):
        return _to_jsonable(x.detach().cpu().tolist())
    return x


def _exact_group_recovery(pred_group_ids, true_group_ids):
    pred_group_ids = np.asarray(pred_group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)
    k = len(pred_group_ids)
    for i in range(k):
        for j in range(i + 1, k):
            pred_same = pred_group_ids[i] == pred_group_ids[j]
            true_same = true_group_ids[i] == true_group_ids[j]
            if pred_same != true_same:
                return False
    return True


def _adjusted_rand_index(pred_group_ids, true_group_ids):
    pred_group_ids = np.asarray(pred_group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)
    pred_labels, pred_inverse = np.unique(pred_group_ids, return_inverse=True)
    true_labels, true_inverse = np.unique(true_group_ids, return_inverse=True)
    contingency = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    for i in range(len(pred_group_ids)):
        contingency[pred_inverse[i], true_inverse[i]] += 1

    sum_comb_cells = float(sum(_comb2(v) for v in contingency.ravel()))
    sum_comb_rows = float(sum(_comb2(v) for v in contingency.sum(axis=1)))
    sum_comb_cols = float(sum(_comb2(v) for v in contingency.sum(axis=0)))
    total_pairs = float(_comb2(len(pred_group_ids)))
    if total_pairs <= 0.0:
        return 1.0

    expected_index = (sum_comb_rows * sum_comb_cols) / total_pairs
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = max_index - expected_index
    if abs(denom) < 1e-12:
        return 1.0
    return float((sum_comb_cells - expected_index) / denom)


def _grouping_metrics(group_ids, true_group_ids):
    group_ids = np.asarray(group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)
    return {
        "group_ids": group_ids.astype(int).tolist(),
        "selected_groups": _group_ids_to_groups(group_ids),
        "n_groups": int(len(np.unique(group_ids))),
        "exact_recovery": bool(_exact_group_recovery(group_ids, true_group_ids)),
        "pairwise_accuracy": float(_pairwise_grouping_accuracy(group_ids, true_group_ids)),
        "ari": float(_adjusted_rand_index(group_ids, true_group_ids)),
    }


def _build_candidate_record(
    *,
    source_name,
    cycle,
    val_mse,
    test_mse,
    n_groups,
    group_ids,
    lambda_index=None,
    lambda_value=None,
    cycle_score=None,
    refit_val_mse=None,
    anchor_gap=None,
    local_gap=None,
    source_detail=None,
    phase1_val_mse=None,
    plots=None,
    grouping_metrics=None,
):
    rec = {
        "source": str(source_name),
        "cycle": (int(cycle) if cycle is not None else None),
        "cycle_score": (float(cycle_score) if cycle_score is not None else None),
        "val_mse": float(val_mse),
        "test_mse": float(test_mse),
        "n_groups": int(n_groups),
        "lambda_index": (int(lambda_index) if lambda_index is not None else None),
        "lambda_value": (float(lambda_value) if lambda_value is not None else None),
        "refit_val_mse": (float(refit_val_mse) if refit_val_mse is not None else None),
        "anchor_gap": (float(anchor_gap) if anchor_gap is not None else None),
        "local_gap": (float(local_gap) if local_gap is not None else None),
        "source_detail": source_detail,
        "phase1_val_mse": (float(phase1_val_mse) if phase1_val_mse is not None else None),
        "plots": plots or {},
        "grouping_metrics": grouping_metrics,
    }
    if group_ids is not None:
        rec["group_ids"] = np.asarray(group_ids).astype(int).tolist()
        rec["selected_groups"] = _group_ids_to_groups(group_ids)
    return rec


def _pick_final_recommended(candidates, mse_rel_tol=0.05, mse_abs_tol=0.0):
    valid = [cand for cand in candidates if cand is not None]
    if not valid:
        return None
    best_val = min(float(cand["val_mse"]) for cand in valid)
    cutoff = max(best_val + float(mse_abs_tol), best_val * (1.0 + float(mse_rel_tol)))
    near_best = [cand for cand in valid if float(cand["val_mse"]) <= cutoff]
    return min(
        near_best,
        key=lambda cand: (
            int(cand["n_groups"]),
            float(cand["val_mse"]),
            float(cand["test_mse"]),
            0 if str(cand["source"]) == "structure_best" else 1,
        ),
    )


def _evaluate_policy_from_model(model, X, A_obs, y_obs, mu_true, a_opt, true_group_ids, group_ids=None):
    with torch.no_grad():
        mu_hat = model.predict_all_treatments(X).detach().cpu().numpy()
    if group_ids is not None:
        group_ids = np.asarray(group_ids).astype(int)
        if mu_hat.ndim != 2:
            raise ValueError(f"Expected mu_hat to be 2D, got shape={mu_hat.shape}")
        n_groups = mu_hat.shape[1]
        if group_ids.min() < 0 or group_ids.max() >= n_groups:
            raise ValueError(
                f"group_ids contain values outside grouped prediction range: "
                f"min={group_ids.min()}, max={group_ids.max()}, n_groups={n_groups}"
            )
        mu_hat = mu_hat[:, group_ids]
    a_hat = np.argmax(mu_hat, axis=1).astype(int)
    mu_true_np = mu_true.detach().cpu().numpy()
    a_opt_np = a_opt.detach().cpu().numpy().astype(int)
    A_obs_np = A_obs.detach().cpu().numpy().astype(int)
    y_obs_np = y_obs.detach().cpu().numpy()
    true_group_ids = np.asarray(true_group_ids).astype(int)

    factual_pred = mu_hat[np.arange(len(A_obs_np)), A_obs_np]
    value = float(mu_true_np[np.arange(len(a_hat)), a_hat].mean())
    oracle_value = float(mu_true_np[np.arange(len(a_opt_np)), a_opt_np].mean())
    regret = float(oracle_value - value)
    acc_opt_action = float((a_hat == a_opt_np).mean())
    pred_group = true_group_ids[a_hat]
    opt_group = true_group_ids[a_opt_np]
    acc_opt_group = float((pred_group == opt_group).mean())
    return {
        "value": value,
        "oracle_value": oracle_value,
        "regret": regret,
        "acc_opt_action": acc_opt_action,
        "mis_rate_action": float(1.0 - acc_opt_action),
        "acc_opt_group": acc_opt_group,
        "mis_rate_group": float(1.0 - acc_opt_group),
        "test_factual_mse": float(np.mean((y_obs_np - factual_pred) ** 2)),
    }


def save_cycle_artifacts(outdir, cycle_id, lambda_values, cycle_pack, args, log_fn):
    """
    cycle_pack must contain:
      - "all_Z": list of (K, r) np arrays
      - "admm_metrics": list[dict]
      - "picked_lambda_index": int or None
      - "refit_results": optional, list[dict]
    """
    Z_list = cycle_pack["all_Z"]
    metrics = cycle_pack["admm_metrics"]
    picked_lambda_index = cycle_pack.get("picked_lambda_index", None)
    refit_results = cycle_pack.get("refit_results", None)

    # 1) save Z_path
    Z_path = np.stack(Z_list, axis=0).astype(np.float32)  # (L, K, r)
    zpath_file = os.path.join(outdir, f"Z_path_cycle{cycle_id}.npy")
    np.save(zpath_file, Z_path)
    log_fn(f"[Save] Z_path -> {zpath_file} | shape={Z_path.shape}")

    # 2) save ADMM path metrics
    metrics_file = os.path.join(outdir, f"path_metrics_cycle{cycle_id}.csv")
    save_path_metrics_csv(metrics, metrics_file)
    log_fn(f"[Save] metrics -> {metrics_file}")

    # 3) dendrogram
    dendro_file = os.path.join(outdir, f"dendrogram_cycle{cycle_id}.png")
    dendro = plot_scaf_dendrogram(
        lambda_values=lambda_values,
        all_Z_list=Z_list,
        round_decimals=int(args.round_decimals),
        title=(
            f"NSCF fusion path, cycle {cycle_id} "
            f"(rho={args.rho})"
        ),
        save_path=dendro_file,
        xlabel="Treatment Index",
        picked_lambda_index=picked_lambda_index,
    )
    log_fn(f"[Save] dendrogram -> {dendro}")

    # 4) refit-based plots
    if refit_results is not None:
        refit_curve_file = os.path.join(outdir, f"refit_lambda_curve_cycle{cycle_id}.png")
        plot_refit_lambda_selection_curve(
            lambda_values=lambda_values,
            refit_results=refit_results,
            picked_lambda_index=picked_lambda_index,
            save_path=refit_curve_file,
            old_metrics=metrics,
        )
        log_fn(f"[Save] refit lambda curve -> {refit_curve_file}")

        refit_mse_file = os.path.join(outdir, f"refit_mse_vs_lambda_cycle{cycle_id}.png")
        plot_refit_mse_vs_lambda(
            lambda_values=lambda_values,
            refit_results=refit_results,
            picked_lambda_index=picked_lambda_index,
            save_path=refit_mse_file,
            old_metrics=metrics,
        )
        log_fn(f"[Save] refit mse curve -> {refit_mse_file}")

        groups_file = os.path.join(outdir, f"groups_vs_lambda_cycle{cycle_id}.png")
        plot_groups_vs_lambda(
            lambda_values=lambda_values,
            refit_results=refit_results,
            picked_lambda_index=picked_lambda_index,
            save_path=groups_file,
        )
        log_fn(f"[Save] groups curve -> {groups_file}")


def set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # -------------------------
    # global seed / deterministic
    # -------------------------
    set_global_seed(
        int(getattr(args, "global_seed", 123)),
        deterministic=bool(getattr(args, "deterministic", False)),
    )

    # -------------------------
    # output / logger
    # -------------------------
    outdir = os.path.join(args.out_base, args.run_name)
    os.makedirs(outdir, exist_ok=True)

    log_fn, log_file = make_logger(os.path.join(outdir, "run.log"))
    log_fn(f"[Run] outdir = {outdir}")
    log_fn(f"[Run] mode = {args.mode}")
    log_fn(
        "[Run] seeds | "
        f"global={int(getattr(args, 'global_seed', 123))} "
        f"train={int(args.seed_train)} val={int(args.seed_val)} test={int(args.seed_test)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"--- Using device: {device} ---")

    # -------------------------
    # Data
    # -------------------------
    K = int(args.n_actions)

    if K != 10:
        raise ValueError("This synthetic currently assumes K=10 treatments (5+5).")

    true_groups = [list(range(0, 5)), list(range(5, 10))]
    true_group_ids = _group_ids_from_groups(true_groups, K)
    log_fn(f"[Truth] true_groups = {true_groups}")

    X_train, A_train, y_train, _ = generate_synthetic_data(
        n_samples=int(args.n_train),
        n_features=int(args.n_features),
        n_treatments=K,
        groups=true_groups,
        seed=int(args.seed_train),
        scenario=str(args.scenario),
        sigma=float(args.y_noise),
        baseline_scale=float(args.baseline_scale),
        effect_scale=float(args.effect_scale),
        near_fused_eps=float(args.near_fused_eps),
        group_effect_style=str(args.group_effect_style),
        baseline_style=str(args.baseline_style),
        treatment_prob_style=str(args.treatment_prob_style),
        group_prototypes=args.group_prototypes,
        treatment_probs=args.treatment_probs,
        ensure_all_treatments=bool(args.ensure_all_treatments),
    )

    X_val, A_val, y_val, _ = generate_synthetic_data(
        n_samples=int(args.n_val),
        n_features=int(args.n_features),
        n_treatments=K,
        groups=true_groups,
        seed=int(args.seed_val),
        scenario=str(args.scenario),
        sigma=float(args.y_noise),
        baseline_scale=float(args.baseline_scale),
        effect_scale=float(args.effect_scale),
        near_fused_eps=float(args.near_fused_eps),
        group_effect_style=str(args.group_effect_style),
        baseline_style=str(args.baseline_style),
        treatment_prob_style=str(args.treatment_prob_style),
        group_prototypes=args.group_prototypes,
        treatment_probs=args.treatment_probs,
        ensure_all_treatments=bool(args.ensure_all_treatments),
    )

    X_test, A_test, y_test, gamma_true_test = generate_synthetic_data(
        n_samples=int(args.n_test),
        n_features=int(args.n_features),
        n_treatments=K,
        groups=true_groups,
        seed=int(args.seed_test),
        scenario=str(args.scenario),
        sigma=float(args.y_noise),
        baseline_scale=float(args.baseline_scale),
        effect_scale=float(args.effect_scale),
        near_fused_eps=float(args.near_fused_eps),
        group_effect_style=str(args.group_effect_style),
        baseline_style=str(args.baseline_style),
        treatment_prob_style=str(args.treatment_prob_style),
        group_prototypes=args.group_prototypes,
        treatment_probs=args.treatment_probs,
        ensure_all_treatments=bool(args.ensure_all_treatments),
    )

    X_train = X_train.to(device)
    A_train = A_train.to(device)
    y_train = y_train.to(device)

    X_val = X_val.to(device)
    A_val = A_val.to(device)
    y_val = y_val.to(device)

    X_test = X_test.to(device)
    A_test = A_test.to(device)
    y_test = y_test.to(device)
    gamma_true_test = gamma_true_test.to(device)
    mu_true_test, a_opt_test = compute_true_mu_and_opt_actions(
        X_test,
        gamma_true_test,
        scenario=str(args.scenario),
        baseline_scale=float(args.baseline_scale),
    )

    # -------------------------
    # Save expected true grouping
    # -------------------------
    true_group_file = os.path.join(outdir, "true_group_structure.png")
    save_true_group_heatmap(
        groups=true_groups,
        n_treatments=K,
        save_path=true_group_file,
        title="Expected True Group Structure",
    )
    log_fn(f"[Save] true grouping -> {true_group_file}")

    # -------------------------
    # Model
    # -------------------------
    # Data generation resets torch RNG through dataset-specific seeds, so we
    # reseed here to make model initialization and training randomness follow
    # the requested global experiment seed.
    set_global_seed(
        int(getattr(args, "global_seed", 123)),
        deterministic=bool(getattr(args, "deterministic", False)),
    )
    model = Net(
        n_features=args.n_features,
        n_treatments=args.n_actions,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        use_baseline=args.use_baseline,
        n_hidden_layers=args.n_hidden_layers,
    ).to(device)

    # -------------------------
    # Lambda path
    # -------------------------
    lambda_values = build_lambda_path(args.num_lambda, args.max_lambda)
    np.save(os.path.join(outdir, "lambda_values.npy"), lambda_values)
    log_fn(f"Lambda values (len={len(lambda_values)}): {lambda_values}")

    t0 = time.time()

    # -------------------------
    # Run alternating only
    # -------------------------
    if args.mode != "alt":
        raise ValueError(f"This cleaned runner only supports mode='alt', got mode={args.mode}")

    alt_out = run_alternating(
        model=model,
        X_train=X_train, A_train=A_train, y_train=y_train,
        X_val=X_val, A_val=A_val, y_val=y_val,
        X_test=X_test, A_test=A_test, y_test=y_test,
        true_group_ids=true_group_ids,
        lambda_values=lambda_values,
        args=args,
        log_fn=log_fn,
        outdir=outdir,
    )

    cycles = alt_out["cycles"]

    if bool(getattr(args, "save_all_cycle_artifacts", False)):
        for pack in cycles:
            cid = int(pack["cycle"])
            save_cycle_artifacts(outdir, cid, lambda_values, pack, args, log_fn)

    cycle_progress_file = os.path.join(outdir, "cycle_progress.png")
    plot_cycle_summary(cycles, cycle_progress_file)
    log_fn(f"[Save] cycle progress -> {cycle_progress_file}")

    best_cycle = int(alt_out["best_cycle_id"])
    best_cycle_pack = cycles[best_cycle - 1]

    final_tradeoff_file = os.path.join(outdir, "prediction_best_lambda_tradeoff.png")
    plot_refit_lambda_selection_curve(
        lambda_values=lambda_values,
        refit_results=best_cycle_pack.get("refit_results", None),
        picked_lambda_index=best_cycle_pack.get("picked_lambda_index", None),
        save_path=final_tradeoff_file,
        old_metrics=best_cycle_pack.get("admm_metrics", None),
    )
    log_fn(f"[Save] prediction-best lambda tradeoff -> {final_tradeoff_file}")

    final_dendro_file = None
    if bool(getattr(args, "save_path_summaries", False)):
        final_dendro_file = os.path.join(outdir, "prediction_best_solution_path_summary.png")
        plot_solution_path_dendrogram(
            lambda_values=lambda_values,
            all_Z_list=best_cycle_pack["all_Z"],
            picked_lambda_index=best_cycle_pack.get("picked_lambda_index", None),
            save_path=final_dendro_file,
            round_decimals=int(args.round_decimals),
            snap_eps=float(args.snap_eps) if getattr(args, "snap_eps", None) is not None else None,
            labels=[str(i + 1) for i in range(args.n_actions)],
            xlabel="Treatment",
            ylabel=r"$\log(\lambda \times 1000)$",
            title=f"Prediction-Best Cycle {best_cycle} Path Summary",
            lambda_scale=1000.0,
        )
        log_fn(f"[Save] prediction-best path summary -> {final_dendro_file}")

    # use the true global best returned by run_alternating
    best_cycle_score = float(alt_out["best_cycle_score"])
    best_cycle_val_mse = float(alt_out["best_val_mse"])
    
    best_picked_lambda_index = alt_out.get("best_picked_lambda_index", None)
    best_picked_lambda_value = alt_out.get("best_picked_lambda_value", None)
    best_picked_n_groups = alt_out.get("best_picked_n_groups", None)
    best_picked_refit_val_mse = alt_out.get("best_picked_refit_val_mse", None)
    pretrain_val_mse = alt_out.get("pretrain_val_mse", None)
    anchor_val_mse = alt_out.get("anchor_val_mse", None)
    best_cycle_beats_anchor = alt_out.get("best_cycle_beats_anchor", None)

    best_alt_cycle = alt_out.get("best_alt_cycle_id", None)
    best_alt_cycle_score = alt_out.get("best_alt_cycle_score", None)
    best_alt_cycle_val_mse = alt_out.get("best_alt_cycle_val_mse", None)
    best_alt_picked_lambda_index = alt_out.get("best_alt_picked_lambda_index", None)
    structure_best_rec = alt_out.get("structure_best_cycle_record", None)

    best_model = alt_out.get("best_model", model)
    best_group_ids = alt_out.get("best_group_ids", None)
    oracle_true_group = alt_out.get("oracle_true_group_refit", None)
    two_group_polish = alt_out.get("two_group_polish", None)

    prediction_best_group_ids = alt_out.get("best_group_ids", None)
    prediction_partition_file = None
    prediction_partition_dendro_file = None
    prediction_path_groupbars_file = None
    if prediction_best_group_ids is not None:
        prediction_partition_file = os.path.join(outdir, "prediction_best_selected_partition.png")
        save_partition_heatmap_from_group_ids(
            prediction_best_group_ids,
            prediction_partition_file,
            title=(
                f"Prediction-Best Selected Partition | cycle={best_cycle}, "
                f"lambda_idx={best_picked_lambda_index}, "
                f"groups={best_picked_n_groups}"
            ),
        )
        log_fn(f"[Save] prediction-best selected partition -> {prediction_partition_file}")
        prediction_partition_dendro_file = os.path.join(
            outdir, "prediction_best_selected_refit_dendrogram.png"
        )
        save_selected_partition_dendrogram_from_group_ids(
            prediction_best_group_ids,
            prediction_partition_dendro_file,
            title=(
                f"Prediction-Best Refit Dendrogram | cycle={best_cycle}, "
                f"lambda={best_picked_lambda_value}, "
                f"groups={best_picked_n_groups}"
            ),
        )
        log_fn(f"[Save] prediction-best refit dendrogram -> {prediction_partition_dendro_file}")
        prediction_path_groupbars_file = os.path.join(
            outdir, "prediction_best_cycle_path_with_groups.png"
        )
        plot_selected_cycle_path_with_group_bars(
            lambda_values=lambda_values,
            all_Z_list=best_cycle_pack["all_Z"],
            picked_lambda_index=best_picked_lambda_index,
            selected_group_ids=prediction_best_group_ids,
            true_group_ids=true_group_ids,
            save_path=prediction_path_groupbars_file,
            round_decimals=int(args.round_decimals),
            snap_eps=float(args.snap_eps) if getattr(args, "snap_eps", None) is not None else None,
            labels=[str(i + 1) for i in range(args.n_actions)],
            title=(
                f"Prediction-Best Cycle {best_cycle} | "
                f"λ={best_picked_lambda_value:.4g}, groups={best_picked_n_groups}"
            ),
        )
        log_fn(f"[Save] prediction-best path-with-groups -> {prediction_path_groupbars_file}")

    oracle_partition_file = None
    oracle_partition_dendro_file = None
    if oracle_true_group is not None:
        oracle_group_ids = np.asarray(oracle_true_group["group_ids"]).astype(int)
        oracle_partition_file = os.path.join(outdir, "oracle_true_2group_partition.png")
        save_partition_heatmap_from_group_ids(
            oracle_group_ids,
            oracle_partition_file,
            title=(
                "Oracle True 2-Group Partition | "
                f"val={float(oracle_true_group['val_mse']):.4f}, "
                f"test={float(oracle_true_group['test_mse']):.4f}"
            ),
        )
        log_fn(f"[Save] oracle true-2-group partition -> {oracle_partition_file}")
        oracle_partition_dendro_file = os.path.join(
            outdir, "oracle_true_2group_refit_dendrogram.png"
        )
        save_selected_partition_dendrogram_from_group_ids(
            oracle_group_ids,
            oracle_partition_dendro_file,
            title=(
                "Oracle True 2-Group Refit Dendrogram | "
                f"val={float(oracle_true_group['val_mse']):.4f}, "
                f"test={float(oracle_true_group['test_mse']):.4f}"
            ),
        )
        log_fn(f"[Save] oracle true-2-group refit dendrogram -> {oracle_partition_dendro_file}")

    two_group_polish_partition_file = None
    two_group_polish_partition_dendro_file = None
    if two_group_polish is not None:
        polish_group_ids = np.asarray(two_group_polish["group_ids"]).astype(int)
        two_group_polish_partition_file = os.path.join(outdir, "two_group_polish_partition.png")
        save_partition_heatmap_from_group_ids(
            polish_group_ids,
            two_group_polish_partition_file,
            title=(
                "Best Data-Driven Two-Group Polish | "
                f"candidate={two_group_polish.get('candidate_name')}, "
                f"source={two_group_polish['source_name']}, "
                f"val={float(two_group_polish['val_mse']):.4f}, "
                f"test={float(two_group_polish['test_mse']):.4f}"
            ),
        )
        log_fn(f"[Save] two-group polish partition -> {two_group_polish_partition_file}")
        two_group_polish_partition_dendro_file = os.path.join(
            outdir, "two_group_polish_refit_dendrogram.png"
        )
        save_selected_partition_dendrogram_from_group_ids(
            polish_group_ids,
            two_group_polish_partition_dendro_file,
            title=(
                "Best Data-Driven Two-Group Polish Dendrogram | "
                f"candidate={two_group_polish.get('candidate_name')}, "
                f"source={two_group_polish['source_name']}, "
                f"val={float(two_group_polish['val_mse']):.4f}, "
                f"test={float(two_group_polish['test_mse']):.4f}"
            ),
        )
        log_fn(f"[Save] two-group polish refit dendrogram -> {two_group_polish_partition_dendro_file}")

    # final mse using the same grouped solution that won cycle selection
    if best_group_ids is not None:
        test_mse = eval_grouped_model_on_original_actions(
            best_model,
            eval_mse,
            X_test,
            A_test,
            y_test,
            best_group_ids,
        )
    else:
        test_mse = eval_mse(model, X_test, A_test, y_test)
    log_fn(f"[Result] final test_mse = {test_mse:.6f}")

    structure_partition_file = None
    structure_partition_dendro_file = None
    structure_dendro_file = None
    structure_tradeoff_file = None
    structure_path_groupbars_file = None
    structure_test_mse = None
    structure_group_ids = None
    if structure_best_rec is not None:
        structure_cycle = int(structure_best_rec["cycle"])
        structure_cycle_pack = cycles[structure_cycle - 1]
        structure_group_ids = structure_best_rec.get("picked_group_ids", None)
        is_same_selection = (
            structure_cycle == best_cycle and
            int(structure_best_rec.get("picked_lambda_index")) == int(best_cycle_pack.get("picked_lambda_index"))
        )

        if not is_same_selection:
            structure_tradeoff_file = os.path.join(outdir, "structure_best_lambda_tradeoff.png")
            plot_refit_lambda_selection_curve(
                lambda_values=lambda_values,
                refit_results=structure_cycle_pack.get("refit_results", None),
                picked_lambda_index=structure_cycle_pack.get("picked_lambda_index", None),
                save_path=structure_tradeoff_file,
                old_metrics=structure_cycle_pack.get("admm_metrics", None),
            )
            log_fn(f"[Save] structure-best lambda tradeoff -> {structure_tradeoff_file}")

            if bool(getattr(args, "save_path_summaries", False)):
                structure_dendro_file = os.path.join(outdir, "structure_best_solution_path_summary.png")
                plot_solution_path_dendrogram(
                    lambda_values=lambda_values,
                    all_Z_list=structure_cycle_pack["all_Z"],
                    picked_lambda_index=structure_cycle_pack.get("picked_lambda_index", None),
                    save_path=structure_dendro_file,
                    round_decimals=int(args.round_decimals),
                    snap_eps=float(args.snap_eps) if getattr(args, "snap_eps", None) is not None else None,
                    labels=[str(i + 1) for i in range(args.n_actions)],
                    xlabel="Treatment",
                    ylabel=r"$\log(\lambda \times 1000)$",
                    title=f"Structure-Best Cycle {structure_cycle} Path Summary",
                    lambda_scale=1000.0,
                )
                log_fn(f"[Save] structure-best path summary -> {structure_dendro_file}")

        if structure_group_ids is not None:
            if not is_same_selection:
                structure_partition_file = os.path.join(outdir, "structure_best_selected_partition.png")
                save_partition_heatmap_from_group_ids(
                    structure_group_ids,
                    structure_partition_file,
                    title=(
                        f"Structure-Best Selected Partition | cycle={structure_cycle}, "
                        f"lambda_idx={structure_cycle_pack.get('picked_lambda_index')}, "
                        f"groups={structure_cycle_pack.get('picked_n_groups')}"
                    ),
                )
                log_fn(f"[Save] structure-best selected partition -> {structure_partition_file}")
                structure_partition_dendro_file = os.path.join(
                    outdir, "structure_best_selected_refit_dendrogram.png"
                )
                save_selected_partition_dendrogram_from_group_ids(
                    structure_group_ids,
                    structure_partition_dendro_file,
                    title=(
                        f"Structure-Best Refit Dendrogram | cycle={structure_cycle}, "
                        f"lambda={structure_cycle_pack.get('picked_lambda_value', structure_best_rec.get('picked_lambda_value'))}, "
                        f"groups={structure_cycle_pack.get('picked_n_groups')}"
                    ),
                )
                log_fn(f"[Save] structure-best refit dendrogram -> {structure_partition_dendro_file}")
                structure_path_groupbars_file = os.path.join(
                    outdir, "structure_best_cycle_path_with_groups.png"
                )
                plot_selected_cycle_path_with_group_bars(
                    lambda_values=lambda_values,
                    all_Z_list=structure_cycle_pack["all_Z"],
                    picked_lambda_index=int(structure_best_rec.get("picked_lambda_index")),
                    selected_group_ids=structure_group_ids,
                    true_group_ids=true_group_ids,
                    save_path=structure_path_groupbars_file,
                    round_decimals=int(args.round_decimals),
                    snap_eps=float(args.snap_eps) if getattr(args, "snap_eps", None) is not None else None,
                    labels=[str(i + 1) for i in range(args.n_actions)],
                    title=(
                        f"Structure-Best Cycle {structure_cycle} | "
                        f"λ={float(structure_best_rec.get('picked_lambda_value')):.4g}, "
                        f"groups={int(structure_best_rec.get('picked_n_groups'))}"
                    ),
                )
                log_fn(f"[Save] structure-best path-with-groups -> {structure_path_groupbars_file}")

        structure_model = structure_best_rec.get("grouped_model", None)
        if structure_model is not None and structure_group_ids is not None:
            structure_test_mse = eval_grouped_model_on_original_actions(
                structure_model,
                eval_mse,
                X_test,
                A_test,
                y_test,
                structure_group_ids,
            )
            log_fn(f"[Result] structure-best test_mse = {structure_test_mse:.6f}")

    prediction_grouping = (
        _grouping_metrics(prediction_best_group_ids, true_group_ids)
        if prediction_best_group_ids is not None else None
    )
    oracle_grouping = (
        _grouping_metrics(oracle_true_group["group_ids"], true_group_ids)
        if oracle_true_group is not None else None
    )
    structure_grouping = (
        _grouping_metrics(structure_group_ids, true_group_ids)
        if structure_group_ids is not None else None
    )

    prediction_candidate = _build_candidate_record(
        source_name="prediction_best",
        cycle=best_cycle,
        cycle_score=best_cycle_score,
        val_mse=best_cycle_val_mse,
        test_mse=float(test_mse),
        n_groups=best_picked_n_groups,
        group_ids=prediction_best_group_ids,
        lambda_index=best_picked_lambda_index,
        lambda_value=best_picked_lambda_value,
        refit_val_mse=best_picked_refit_val_mse,
        plots={
            "lambda_tradeoff_plot": final_tradeoff_file,
            "path_summary_plot": final_dendro_file,
            "selected_partition_plot": prediction_partition_file,
            "selected_refit_dendrogram_plot": prediction_partition_dendro_file,
            "path_with_groups_plot": prediction_path_groupbars_file,
        },
        grouping_metrics=prediction_grouping,
    )
    structure_candidate = None
    if structure_best_rec is not None:
        structure_candidate = _build_candidate_record(
            source_name="structure_best",
            cycle=int(structure_best_rec["cycle"]),
            cycle_score=float(structure_best_rec["cycle_score"]),
            val_mse=float(structure_best_rec["cycle_val_mse"]),
            test_mse=(
                float(structure_test_mse) if structure_test_mse is not None else float(test_mse)
            ),
            n_groups=int(structure_best_rec["picked_n_groups"]),
            group_ids=structure_group_ids,
            lambda_index=int(structure_best_rec["picked_lambda_index"]),
            lambda_value=float(structure_best_rec["picked_lambda_value"]),
            anchor_gap=float(structure_best_rec["anchor_gap"]),
            local_gap=float(structure_best_rec["local_gap"]),
            plots={
                "lambda_tradeoff_plot": (
                    structure_tradeoff_file if structure_tradeoff_file is not None else final_tradeoff_file
                ),
                "path_summary_plot": (
                    structure_dendro_file if structure_dendro_file is not None else final_dendro_file
                ),
                "selected_partition_plot": (
                    structure_partition_file if structure_partition_file is not None else prediction_partition_file
                ),
                "selected_refit_dendrogram_plot": (
                    structure_partition_dendro_file
                    if structure_partition_dendro_file is not None
                    else prediction_partition_dendro_file
                ),
                "path_with_groups_plot": (
                    structure_path_groupbars_file
                    if structure_path_groupbars_file is not None
                    else prediction_path_groupbars_file
                ),
            },
            grouping_metrics=structure_grouping,
        )
    final_recommended = _pick_final_recommended(
        [prediction_candidate, structure_candidate],
        mse_rel_tol=float(getattr(args, "structure_mse_rel_tol", 0.05)),
        mse_abs_tol=float(getattr(args, "structure_mse_abs_tol", 0.0)),
    )
    final_recommended_partition_file = None
    final_recommended_partition_dendro_file = None
    final_recommended_path_groupbars_file = None
    if final_recommended is not None and final_recommended.get("group_ids") is not None:
        final_group_ids = np.asarray(final_recommended["group_ids"]).astype(int)
        final_recommended_partition_file = os.path.join(outdir, "final_recommended_partition.png")
        save_partition_heatmap_from_group_ids(
            final_group_ids,
            final_recommended_partition_file,
            title=(
                f"Final Recommended Partition | source={final_recommended['source']}, "
                f"val={float(final_recommended['val_mse']):.4f}, "
                f"test={float(final_recommended['test_mse']):.4f}, "
                f"groups={int(final_recommended['n_groups'])}"
            ),
        )
        log_fn(f"[Save] final recommended partition -> {final_recommended_partition_file}")

        final_recommended_partition_dendro_file = os.path.join(
            outdir, "final_recommended_refit_dendrogram.png"
        )
        save_selected_partition_dendrogram_from_group_ids(
            final_group_ids,
            final_recommended_partition_dendro_file,
            title=(
                f"Final Recommended Dendrogram | source={final_recommended['source']}, "
                f"groups={int(final_recommended['n_groups'])}"
            ),
        )
        log_fn(
            f"[Save] final recommended refit dendrogram -> {final_recommended_partition_dendro_file}"
        )

        ref_cycle = best_cycle
        ref_cycle_pack = best_cycle_pack
        if str(final_recommended["source"]) == "structure_best" and structure_best_rec is not None:
            ref_cycle = int(structure_best_rec["cycle"])
            ref_cycle_pack = cycles[ref_cycle - 1]
        final_recommended_path_groupbars_file = os.path.join(
            outdir, "final_recommended_path_with_groups.png"
        )
        plot_selected_cycle_path_with_group_bars(
            lambda_values=lambda_values,
            all_Z_list=ref_cycle_pack["all_Z"],
            picked_lambda_index=(
                int(final_recommended["lambda_index"])
                if final_recommended.get("lambda_index") is not None
                else None
            ),
            selected_group_ids=final_group_ids,
            true_group_ids=true_group_ids,
            save_path=final_recommended_path_groupbars_file,
            round_decimals=int(args.round_decimals),
            snap_eps=float(args.snap_eps) if getattr(args, "snap_eps", None) is not None else None,
            labels=[str(i + 1) for i in range(args.n_actions)],
            title=(
                f"Final Recommended | source={final_recommended['source']}, "
                f"cycle={ref_cycle}, groups={int(final_recommended['n_groups'])}"
            ),
        )
        log_fn(f"[Save] final recommended path-with-groups -> {final_recommended_path_groupbars_file}")
        final_recommended["plots"] = dict(final_recommended.get("plots", {}))
        final_recommended["plots"]["selected_partition_plot"] = final_recommended_partition_file
        final_recommended["plots"]["selected_refit_dendrogram_plot"] = (
            final_recommended_partition_dendro_file
        )
        final_recommended["plots"]["path_with_groups_plot"] = final_recommended_path_groupbars_file
        log_fn(
            "[Final] recommended | "
            f"source={final_recommended['source']}, "
            f"val={float(final_recommended['val_mse']):.6f}, "
            f"test={float(final_recommended['test_mse']):.6f}, "
            f"groups={int(final_recommended['n_groups'])}"
        )

    prediction_policy = None
    prediction_model = best_cycle_pack.get("grouped_model", None)
    if prediction_model is not None:
        prediction_policy = _evaluate_policy_from_model(
            prediction_model,
            X_test,
            A_test,
            y_test,
            mu_true_test,
            a_opt_test,
            true_group_ids,
            group_ids=prediction_best_group_ids,
        )

    structure_policy = None
    if structure_model is not None:
        structure_policy = _evaluate_policy_from_model(
            structure_model,
            X_test,
            A_test,
            y_test,
            mu_true_test,
            a_opt_test,
            true_group_ids,
            group_ids=structure_group_ids,
        )

    oracle_policy = None
    oracle_model = oracle_true_group.get("model", None) if oracle_true_group is not None else None
    if oracle_model is not None:
        oracle_policy = _evaluate_policy_from_model(
            oracle_model,
            X_test,
            A_test,
            y_test,
            mu_true_test,
            a_opt_test,
            true_group_ids,
            group_ids=oracle_true_group.get("group_ids"),
        )

    final_policy = None
    if final_recommended is not None:
        final_model = None
        if str(final_recommended["source"]) == "prediction_best":
            final_model = prediction_model
        elif str(final_recommended["source"]) == "structure_best":
            final_model = structure_model
        if final_model is not None:
            final_group_ids = final_recommended.get("group_ids")
            final_policy = _evaluate_policy_from_model(
                final_model,
                X_test,
                A_test,
                y_test,
                mu_true_test,
                a_opt_test,
                true_group_ids,
                group_ids=final_group_ids,
            )

    summary = {
        "run": {
            "mode": "alt",
            "n_cycles_completed": len(cycles),
            "outdir": outdir,
            "cycle_progress_plot": cycle_progress_file,
        },
        "truth": {
            "true_groups": true_groups,
            "true_group_ids": true_group_ids.astype(int).tolist(),
        },
        "baseline": {
            "pretrain_val_mse": pretrain_val_mse,
            "anchor_val_mse": anchor_val_mse,
        },
        "prediction_best": {
            "cycle": best_cycle,
            "cycle_score": best_cycle_score,
            "val_mse": best_cycle_val_mse,
            "test_mse": float(test_mse),
            "beats_anchor": best_cycle_beats_anchor,
            "lambda_index": best_picked_lambda_index,
            "lambda_value": best_picked_lambda_value,
            "n_groups": best_picked_n_groups,
            "refit_val_mse": best_picked_refit_val_mse,
            "lambda_tradeoff_plot": final_tradeoff_file,
            "path_summary_plot": final_dendro_file,
            "selected_partition_plot": prediction_partition_file,
            "selected_refit_dendrogram_plot": prediction_partition_dendro_file,
            "path_with_groups_plot": prediction_path_groupbars_file,
            "grouping_metrics": prediction_grouping,
            "selected_groups": (
                prediction_grouping["selected_groups"] if prediction_grouping is not None else None
            ),
            "group_ids": (
                prediction_grouping["group_ids"] if prediction_grouping is not None else None
            ),
            "policy_metrics": prediction_policy,
        },
        "selection_debug": {
            "best_alt_cycle": best_alt_cycle,
            "best_alt_cycle_score": best_alt_cycle_score,
            "best_alt_cycle_val_mse": best_alt_cycle_val_mse,
            "best_alt_picked_lambda_index": best_alt_picked_lambda_index,
        },
    }

    if oracle_true_group is not None:
        oracle_gap_vs_prediction_best = (
            float(oracle_true_group["val_mse"]) - float(best_cycle_val_mse)
        )
        oracle_gap_vs_anchor = (
            float(oracle_true_group["val_mse"]) - float(anchor_val_mse)
        )
        summary["oracle_true_2group"] = {
            "groups": true_groups,
            "n_groups": int(oracle_true_group["n_groups"]),
            "val_mse": float(oracle_true_group["val_mse"]),
            "test_mse": float(oracle_true_group["test_mse"]),
            "val_gap_vs_prediction_best": float(oracle_gap_vs_prediction_best),
            "val_gap_vs_anchor": float(oracle_gap_vs_anchor),
            "selected_partition_plot": oracle_partition_file,
            "selected_refit_dendrogram_plot": oracle_partition_dendro_file,
            "diagnosis": _oracle_diagnosis(
                best_cycle_val_mse, oracle_true_group["val_mse"]
            ),
            "grouping_metrics": oracle_grouping,
            "selected_groups": (
                oracle_grouping["selected_groups"] if oracle_grouping is not None else None
            ),
            "group_ids": (
                oracle_grouping["group_ids"] if oracle_grouping is not None else None
            ),
            "policy_metrics": oracle_policy,
        }

    if structure_best_rec is not None:
        summary["structure_best"] = {
            "cycle": int(structure_best_rec["cycle"]),
            "cycle_score": float(structure_best_rec["cycle_score"]),
            "val_mse": float(structure_best_rec["cycle_val_mse"]),
            "test_mse": (
                float(structure_test_mse) if structure_test_mse is not None else None
            ),
            "lambda_index": int(structure_best_rec["picked_lambda_index"]),
            "lambda_value": float(structure_best_rec["picked_lambda_value"]),
            "n_groups": int(structure_best_rec["picked_n_groups"]),
            "anchor_gap": float(structure_best_rec["anchor_gap"]),
            "local_gap": float(structure_best_rec["local_gap"]),
            "lambda_tradeoff_plot": (
                structure_tradeoff_file if structure_tradeoff_file is not None else final_tradeoff_file
            ),
            "path_summary_plot": (
                structure_dendro_file if structure_dendro_file is not None else final_dendro_file
            ),
            "selected_partition_plot": (
                structure_partition_file if structure_partition_file is not None else prediction_partition_file
            ),
            "selected_refit_dendrogram_plot": (
                structure_partition_dendro_file
                if structure_partition_dendro_file is not None
                else prediction_partition_dendro_file
            ),
            "path_with_groups_plot": (
                structure_path_groupbars_file
                if structure_path_groupbars_file is not None
                else prediction_path_groupbars_file
            ),
            "grouping_metrics": structure_grouping,
            "selected_groups": (
                structure_grouping["selected_groups"] if structure_grouping is not None else None
            ),
            "group_ids": (
                structure_grouping["group_ids"] if structure_grouping is not None else None
            ),
            "policy_metrics": structure_policy,
            "rule": (
                "among grouped candidates that still beat anchor/pretrain, "
                "prefer the simplest one within the allowed MSE band near prediction-best"
            ),
            "mse_rel_tol_vs_prediction_best": float(
                getattr(args, "structure_mse_rel_tol", 0.05)
            ),
            "mse_abs_tol_vs_prediction_best": float(
                getattr(args, "structure_mse_abs_tol", 0.0)
            ),
        }

    summary["formal_candidates"] = {
        "prediction_best": prediction_candidate,
        "structure_best": structure_candidate,
    }
    if final_recommended is not None:
        summary["final_recommended"] = {
            **final_recommended,
            "policy_metrics": final_policy,
            "selection_rule": (
                "among formal candidates within the allowed validation-MSE band, "
                "choose the simplest structure; break ties by lower val/test MSE"
            ),
            "selected_partition_plot": final_recommended_partition_file,
            "selected_refit_dendrogram_plot": final_recommended_partition_dendro_file,
            "path_with_groups_plot": final_recommended_path_groupbars_file,
        }

    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(_to_jsonable(summary), f, indent=2)
    log_fn(f"[Save] summary -> {summary_path}")

    log_fn(f"[Done] total_time_sec={time.time() - t0:.2f}")
    log_file.close()


if __name__ == "__main__":
    main()
