import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .refit_selection import (
    eval_grouped_model_on_original_actions,
    evaluate_lambda_path_with_refit,
    map_actions_to_groups,
    pick_lambda_by_refit,
    run_grouped_refit,
    save_refit_metrics_csv,
)
from .solver_admm import run_admm_path
from .trainer import eval_mse, train_supervised


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _enable_gamma_only_for_admm(model):
    _freeze_all(model)
    if hasattr(model, "gamma") and hasattr(model.gamma, "weight"):
        model.gamma.weight.requires_grad = True
    elif hasattr(model, "gamma"):
        for p in model.gamma.parameters():
            p.requires_grad = True


def _clusters_from_group_ids(group_ids):
    group_ids = np.asarray(group_ids).astype(int)
    return tuple(group_ids.tolist())


def _stability_fro(Z_prev: torch.Tensor, Z_curr: torch.Tensor) -> float:
    return float(torch.norm(Z_curr - Z_prev, p="fro").item())


def _stability_labels_from_group_ids(prev_ids, curr_ids) -> float:
    prev_ids = np.asarray(prev_ids).astype(int)
    curr_ids = np.asarray(curr_ids).astype(int)
    k = len(prev_ids)
    disagree = 0
    total = 0
    for i in range(k):
        for j in range(i + 1, k):
            same_prev = prev_ids[i] == prev_ids[j]
            same_curr = curr_ids[i] == curr_ids[j]
            if same_prev != same_curr:
                disagree += 1
            total += 1
    return float(disagree / max(total, 1))


def _compute_cycle_stability(
    *,
    prev_Z_star=None,
    curr_Z_star=None,
    prev_group_ids=None,
    curr_group_ids=None,
    metric="fro",
):
    metric = str(metric).lower()
    if metric == "fro":
        if prev_Z_star is None or curr_Z_star is None:
            return None
        return _stability_fro(prev_Z_star, curr_Z_star)
    if metric == "labels":
        if prev_group_ids is None or curr_group_ids is None:
            return None
        return _stability_labels_from_group_ids(prev_group_ids, curr_group_ids)
    raise ValueError(f"Unknown stability metric={metric}")


def _copy_grouped_solution_into_full_model(full_model, grouped_model, group_ids):
    group_ids = np.asarray(group_ids).astype(int)
    with torch.no_grad():
        if hasattr(full_model, "fc1") and getattr(grouped_model, "fc1", None) is not None:
            full_model.fc1.load_state_dict(grouped_model.fc1.state_dict())
        if hasattr(full_model, "fc2") and getattr(full_model, "fc2", None) is not None:
            if getattr(grouped_model, "fc2", None) is not None:
                full_model.fc2.load_state_dict(grouped_model.fc2.state_dict())
        if getattr(full_model, "m_head", None) is not None and getattr(grouped_model, "m_head", None) is not None:
            full_model.m_head.load_state_dict(grouped_model.m_head.state_dict())

        grouped_gamma = grouped_model.gamma.weight.detach().to(
            device=full_model.gamma.weight.device,
            dtype=full_model.gamma.weight.dtype,
        )
        full_model.gamma.weight.copy_(grouped_gamma[group_ids])


def _enumerate_two_group_merges(group_ids):
    group_ids = np.asarray(group_ids).astype(int)
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    if n_groups < 2:
        return []
    if n_groups == 2:
        return [{
            "candidate_name": "already_two_groups",
            "group_ids": group_ids.copy(),
            "origin_n_groups": 2,
        }]

    group_to_actions = {gid: np.where(group_ids == gid)[0].astype(int) for gid in unique_groups}
    first_gid = int(unique_groups[0])
    remaining = [int(g) for g in unique_groups[1:]]
    out = []
    for mask in range(1 << len(remaining)):
        left_group_ids = [first_gid]
        right_group_ids = []
        for bit, gid in enumerate(remaining):
            if (mask >> bit) & 1:
                left_group_ids.append(gid)
            else:
                right_group_ids.append(gid)
        if not right_group_ids:
            continue
        candidate = np.zeros_like(group_ids)
        for gid in left_group_ids:
            candidate[group_to_actions[gid]] = 0
        for gid in right_group_ids:
            candidate[group_to_actions[gid]] = 1
        out.append({
            "candidate_name": (
                "merge_" +
                "-".join(map(str, sorted(left_group_ids))) +
                "__vs__" +
                "-".join(map(str, sorted(right_group_ids)))
            ),
            "group_ids": candidate.astype(int),
            "origin_n_groups": int(n_groups),
        })
    return out


def plot_pretrain_and_cycles(pretrain_val, all_cycles, outdir):
    xs = [0] + [int(c["cycle"]) for c in all_cycles]
    ys = [float(pretrain_val)] + [float(c["cycle_val_mse"]) for c in all_cycles]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Cycle (0 = Pretrain)")
    plt.ylabel("Validation MSE")
    plt.title("Pretrain + Clean Cycles Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pretrain_cycle_val_mse_curve.png"), dpi=200)
    plt.close()


def _pick_cycle_by_prediction(all_cycles):
    if not all_cycles:
        return None
    j = min(
        range(len(all_cycles)),
        key=lambda i: (
            float(all_cycles[i]["cycle_score"]),
            float(all_cycles[i]["cycle_val_mse"]),
            int(all_cycles[i]["picked_n_groups"]),
            int(all_cycles[i]["cycle"]),
        ),
    )
    return all_cycles[j]


def _pick_cycle_by_structure(
    all_cycles,
    anchor_val,
    prediction_best_val=None,
    improve_tol=0.0,
    mse_rel_tol=0.05,
    mse_abs_tol=0.0,
):
    if not all_cycles:
        return None
    improving = [
        rec for rec in all_cycles
        if float(rec["cycle_val_mse"]) <= float(anchor_val) + float(improve_tol)
    ]
    if not improving:
        return None
    if prediction_best_val is not None:
        prediction_best_val = float(prediction_best_val)
        near_opt_cutoff = max(
            prediction_best_val + float(mse_abs_tol),
            prediction_best_val * (1.0 + float(mse_rel_tol)),
        )
        near_opt = [
            rec for rec in improving
            if float(rec["cycle_val_mse"]) <= near_opt_cutoff
        ]
        if near_opt:
            improving = near_opt
    return min(
        improving,
        key=lambda rec: (
            int(rec["picked_n_groups"]),
            float(rec["cycle_val_mse"]),
            int(rec["cycle"]),
        ),
    )


def _build_model_kwargs_from_args(args):
    keys = [
        "n_features",
        "hidden1",
        "hidden2",
        "dropout",
        "use_baseline",
        "n_hidden_layers",
    ]
    out = {}
    for key in keys:
        if hasattr(args, key):
            out[key] = getattr(args, key)
    return out


def _fit_anchor_model(
    model,
    X_train,
    A_train,
    y_train,
    X_val,
    A_val,
    y_val,
    args,
    log_fn,
    target_scale,
):
    log_fn("--- [Stage 1] Anchor Pretrain NN ---")
    best_val = train_supervised(
        model,
        X_train,
        A_train,
        y_train,
        X_val,
        A_val,
        y_val,
        epochs=int(args.train_epochs),
        lr=float(args.train_lr),
        wd=float(args.train_wd),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
        log_fn=log_fn,
        tether_Z=None,
        mu_tether=0.0,
        loss_type=str(getattr(args, "loss_type", "mse")),
        huber_delta=float(getattr(args, "huber_delta", 1.0)),
        improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
        grad_clip=float(getattr(args, "grad_clip", 5.0)),
        target_scale=float(target_scale),
        evaluate_step0=True,
    )
    return float(best_val)


def _run_oracle_true_group_refit(
    *,
    anchor_model,
    true_group_ids,
    model_kwargs,
    target_scale,
    X_train,
    A_train,
    y_train,
    X_val,
    A_val,
    y_val,
    X_test,
    A_test,
    y_test,
    args,
    log_fn,
):
    oracle = run_grouped_refit(
        model_class=type(anchor_model),
        train_fn=train_supervised,
        eval_fn=eval_mse,
        X_train=X_train,
        A_train=A_train,
        y_train=y_train,
        X_val=X_val,
        A_val=A_val,
        y_val=y_val,
        group_ids=np.asarray(true_group_ids).astype(int),
        model_kwargs=model_kwargs,
        train_kwargs=dict(
            epochs=int(getattr(args, "refit_select_epochs", 120)),
            lr=float(getattr(args, "refit_select_lr", 5e-4)),
            wd=float(getattr(args, "refit_select_wd", 1e-4)),
            patience=int(args.patience),
            eval_every=int(args.eval_every),
            log_fn=log_fn,
            tether_Z=None,
            mu_tether=0.0,
            loss_type=str(getattr(args, "loss_type", "mse")),
            huber_delta=float(getattr(args, "huber_delta", 1.0)),
            improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
            grad_clip=float(getattr(args, "grad_clip", 5.0)),
            target_scale=float(target_scale),
            evaluate_step0=True,
        ),
        model_n_actions_key="n_treatments",
        init_model=anchor_model,
    )
    oracle_test_mse = eval_grouped_model_on_original_actions(
        oracle["model"],
        eval_mse,
        X_test,
        A_test,
        y_test,
        oracle["group_ids"],
    )
    oracle["test_mse"] = float(oracle_test_mse)
    oracle["val_mse"] = float(oracle["val_mse"])
    return oracle


def _run_two_group_polish_search(
    *,
    source_models,
    candidate_partitions,
    model_kwargs,
    target_scale,
    X_train,
    A_train,
    y_train,
    X_val,
    A_val,
    y_val,
    X_test,
    A_test,
    y_test,
    args,
    log_fn,
):
    phase1_epochs = int(getattr(args, "two_group_polish_phase1_epochs", 80))
    phase2_epochs = int(getattr(args, "two_group_polish_phase2_epochs", 120))
    phase1_lr = float(getattr(args, "two_group_polish_phase1_lr", getattr(args, "refit_select_lr", 5e-4)))
    phase2_lr = float(getattr(args, "two_group_polish_phase2_lr", getattr(args, "refit_select_lr", 5e-4) * 0.5))
    wd = float(getattr(args, "refit_select_wd", 1e-4))

    results = []
    for cand in candidate_partitions:
        cand_group_ids = np.asarray(cand["group_ids"]).astype(int)
        A_train_g = map_actions_to_groups(A_train, cand_group_ids)
        A_val_g = map_actions_to_groups(A_val, cand_group_ids)
        A_test_g = map_actions_to_groups(A_test, cand_group_ids)
        for source_name, source_model in source_models:
            phase1 = run_grouped_refit(
                model_class=type(source_model),
                train_fn=train_supervised,
                eval_fn=eval_mse,
                X_train=X_train,
                A_train=A_train,
                y_train=y_train,
                X_val=X_val,
                A_val=A_val,
                y_val=y_val,
                group_ids=cand_group_ids,
                model_kwargs=model_kwargs,
                train_kwargs=dict(
                    epochs=phase1_epochs,
                    lr=phase1_lr,
                    wd=wd,
                    patience=int(args.patience),
                    eval_every=int(args.eval_every),
                    log_fn=log_fn,
                    tether_Z=None,
                    mu_tether=0.0,
                    loss_type=str(getattr(args, "loss_type", "mse")),
                    huber_delta=float(getattr(args, "huber_delta", 1.0)),
                    improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
                    grad_clip=float(getattr(args, "grad_clip", 5.0)),
                    target_scale=float(target_scale),
                    evaluate_step0=True,
                    train_backbone=False,
                    train_gamma=True,
                    train_baseline=True,
                ),
                model_n_actions_key="n_treatments",
                init_model=source_model,
            )
            phase1_model = phase1["model"]
            phase2_val = train_supervised(
                phase1_model,
                X_train,
                A_train_g,
                y_train,
                X_val,
                A_val_g,
                y_val,
                epochs=phase2_epochs,
                lr=phase2_lr,
                wd=wd,
                patience=int(args.patience),
                eval_every=int(args.eval_every),
                log_fn=log_fn,
                tether_Z=None,
                mu_tether=0.0,
                loss_type=str(getattr(args, "loss_type", "mse")),
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
                improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
                grad_clip=float(getattr(args, "grad_clip", 5.0)),
                target_scale=float(target_scale),
                evaluate_step0=True,
                train_backbone=True,
                train_gamma=True,
                train_baseline=True,
            )
            phase2_test = float(eval_mse(phase1_model, X_test, A_test_g, y_test))
            results.append({
                "source_name": source_name,
                "candidate_name": cand["candidate_name"],
                "candidate_origin_n_groups": int(cand["origin_n_groups"]),
                "model": phase1_model,
                "group_ids": cand_group_ids.copy(),
                "n_groups": 2,
                "phase1_val_mse": float(phase1["val_mse"]),
                "val_mse": float(phase2_val),
                "test_mse": float(phase2_test),
            })
            log_fn(
                "[Clean] Two-group polish search | "
                f"candidate={cand['candidate_name']}, source={source_name}, "
                f"phase1_val={float(phase1['val_mse']):.6f}, "
                f"final_val={float(phase2_val):.6f}, final_test={float(phase2_test):.6f}"
            )

    best = min(results, key=lambda r: (float(r["val_mse"]), float(r["test_mse"])))
    best["all_results"] = results
    return best


def run_alternating(
    model,
    X_train,
    A_train,
    y_train,
    X_val,
    A_val,
    y_val,
    X_test,
    A_test,
    y_test,
    true_group_ids,
    lambda_values,
    args,
    log_fn=print,
    outdir=None,
):
    """
    Clean workflow:
      1) fit a fixed anchor model
      2) keep a separate working model for cycle-wise optimization
      3) in each cycle, evaluate lambda path by grouped refits initialized from
         the cycle-start snapshot (not the ADMM-mutated model)
      4) select best lambda using validation performance under constrained refit
      5) select best cycle among grouped candidates with a fixed anchor baseline
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    y_train_std = float(torch.std(y_train.view(-1), unbiased=False).item())
    if bool(getattr(args, "scale_loss_by_y_std", True)):
        target_scale = max(float(getattr(args, "min_target_scale", 1.0)), y_train_std)
    else:
        target_scale = 1.0
    log_fn(
        f"[Clean] target_scale_for_loss={target_scale:.6f} "
        f"(y_train_std={y_train_std:.6f})"
    )

    pretrain_val = _fit_anchor_model(
        model,
        X_train,
        A_train,
        y_train,
        X_val,
        A_val,
        y_val,
        args,
        log_fn,
        target_scale,
    )

    anchor_model = copy.deepcopy(model)
    anchor_val = float(pretrain_val)
    working_model = copy.deepcopy(anchor_model)

    best_cycle_id = 0
    best_cycle_score = float("inf")
    best_val = float("inf")
    best_model = copy.deepcopy(anchor_model)
    best_group_ids = np.arange(int(getattr(args, "n_actions")), dtype=int)
    best_cycle_beats_anchor = False

    prev_Z_star = None
    prev_group_ids = None
    all_cycles = []
    no_improve = 0

    Z_init = None
    U_init = None

    refit_select_epochs = int(getattr(args, "refit_select_epochs", 120))
    refit_select_lr = float(getattr(args, "refit_select_lr", getattr(args, "train_lr", 1e-3)))
    refit_select_wd = float(getattr(args, "refit_select_wd", getattr(args, "train_wd", 1e-4)))
    refit_lambda_rule = str(getattr(args, "refit_lambda_rule", "improve_eps_simplest")).lower()
    refit_eps = float(getattr(args, "refit_eps", 0.05))
    refit_group_penalty = float(getattr(args, "refit_group_penalty", 0.0))
    refit_improve_tol = float(getattr(args, "refit_improve_tol", 0.0))
    refit_partition_tol = float(getattr(args, "refit_partition_tol", 1e-6))
    refit_round_decimals = int(getattr(args, "refit_round_decimals", 6))
    dedup_partitions = bool(getattr(args, "dedup_partitions", True))

    cycle_metric = str(getattr(args, "cycle_metric", "val_plus_stability")).lower()
    cycle_improve_tol = float(getattr(args, "cycle_improve_tol", 1e-4))
    cycle_patience = int(getattr(args, "cycle_patience", 2))
    disable_cycle_early_stop = bool(getattr(args, "disable_cycle_early_stop", False))
    stability_metric = str(getattr(args, "stability_metric", "fro")).lower()
    stability_weight = float(getattr(args, "stability_weight", 0.0))
    stability_tol = float(getattr(args, "stability_tol", 1e-3))
    warm_start_from_Zstar = bool(getattr(args, "warm_start_from_Zstar", True))
    reset_U_each_cycle = bool(getattr(args, "reset_U_each_cycle", True))
    save_cycle_summary = bool(getattr(args, "save_cycle_summary", False))

    model_kwargs = _build_model_kwargs_from_args(args)

    for c in range(int(args.alt_cycles)):
        cycle_id = c + 1
        log_fn(f"\n=== [Clean Cycle {cycle_id}/{int(args.alt_cycles)}] ADMM ===")

        cycle_start_model = copy.deepcopy(working_model)
        cycle_start_val = float(eval_mse(cycle_start_model, X_val, A_val, y_val))
        log_fn(
            f"[Clean] Cycle={cycle_id} start working-model val_mse={cycle_start_val:.6f} | "
            f"anchor_pretrain_val_mse={anchor_val:.6f}"
        )

        _enable_gamma_only_for_admm(working_model)
        with torch.no_grad():
            working_model.eval()
            H_train = working_model.extract_features(X_train).detach()
            H_val = working_model.extract_features(X_val).detach()

        all_Z, metrics, Z_final, U_final = run_admm_path(
            model=working_model,
            H_train=H_train,
            A_train=A_train,
            y_train=y_train,
            lambda_values=lambda_values,
            args=args,
            H_val=H_val,
            A_val=A_val,
            y_val=y_val,
            init_Z=Z_init,
            init_U=U_init,
            log_fn=log_fn,
            outdir=outdir,
            save_tag=f"cycle{cycle_id}",
            save_last_ZU=True,
        )

        refit_results = evaluate_lambda_path_with_refit(
            Z_path=np.asarray(all_Z),
            lambda_values=lambda_values,
            model_class=type(working_model),
            train_fn=train_supervised,
            eval_fn=eval_mse,
            X_train=X_train,
            A_train=A_train,
            y_train=y_train,
            X_val=X_val,
            A_val=A_val,
            y_val=y_val,
            model_kwargs=model_kwargs,
            train_kwargs=dict(
                epochs=refit_select_epochs,
                lr=refit_select_lr,
                wd=refit_select_wd,
                patience=int(args.patience),
                eval_every=int(args.eval_every),
                log_fn=log_fn,
                tether_Z=None,
                mu_tether=0.0,
                loss_type=str(getattr(args, "loss_type", "mse")),
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
                improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
                grad_clip=float(getattr(args, "grad_clip", 5.0)),
                target_scale=float(target_scale),
                evaluate_step0=True,
            ),
            init_model=cycle_start_model,
            partition_tol=refit_partition_tol,
            round_decimals=refit_round_decimals,
            dedup_partitions=dedup_partitions,
            model_n_actions_key="n_treatments",
            baseline_mse=anchor_val,
            verbose=True,
        )

        forced_lambda_index = int(getattr(args, "force_refit_lambda_index", -1))
        if forced_lambda_index >= 0:
            if forced_lambda_index >= len(refit_results):
                raise ValueError(
                    "force_refit_lambda_index="
                    f"{forced_lambda_index} is outside the lambda path "
                    f"(length={len(refit_results)})."
                )
            best_i = forced_lambda_index
            log_fn(
                "[Clean] Using externally selected lambda index "
                f"{best_i} for this full-data refit cycle."
            )
        else:
            best_i = pick_lambda_by_refit(
                refit_results,
                rule=refit_lambda_rule,
                eps=refit_eps,
                alpha=refit_group_penalty,
                baseline_mse=anchor_val,
                improve_tol=refit_improve_tol,
            )
        best_refit = refit_results[best_i]
        grouped_model = best_refit["refit_model"]
        picked_group_ids = np.asarray(best_refit["group_ids"]).astype(int)
        Z_star = torch.tensor(
            all_Z[best_i],
            device=X_train.device,
            dtype=working_model.gamma.weight.dtype,
        )

        final_val = float(best_refit["refit_val_mse"])
        anchor_gap = final_val - anchor_val
        local_gap = final_val - cycle_start_val
        full_val = eval_grouped_model_on_original_actions(
            grouped_model,
            eval_mse,
            X_val,
            A_val,
            y_val,
            picked_group_ids,
        )

        log_fn(
            f"[Clean] lambda pick -> idx={best_i}, "
            f"lambda={float(lambda_values[best_i]):.6g}, "
            f"n_groups={int(best_refit['n_groups'])}, "
            f"refit_val_mse={final_val:.6f}, "
            f"anchor_gap={anchor_gap:.6f}, "
            f"local_gap={local_gap:.6f}, "
            f"original_action_eval={full_val:.6f}"
        )

        if outdir is not None:
            save_refit_metrics_csv(
                refit_results,
                os.path.join(outdir, f"refit_metrics_cycle{cycle_id}.csv"),
            )

        stability_to_prev = _compute_cycle_stability(
            prev_Z_star=prev_Z_star,
            curr_Z_star=Z_star,
            prev_group_ids=prev_group_ids,
            curr_group_ids=picked_group_ids,
            metric=stability_metric,
        )
        if stability_to_prev is not None:
            log_fn(
                f"[Clean] Cycle={cycle_id} stability_to_prev "
                f"({stability_metric}) = {stability_to_prev:.6f}"
            )

        if cycle_metric == "val":
            cycle_score = final_val
        elif cycle_metric == "val_plus_stability":
            cycle_score = final_val
            if stability_to_prev is not None:
                cycle_score += stability_weight * stability_to_prev
        else:
            raise ValueError(f"Unknown cycle_metric={cycle_metric}")

        log_fn(f"[Clean] Cycle={cycle_id} cycle_score={cycle_score:.6f}")

        improved = cycle_score < best_cycle_score - cycle_improve_tol
        if improved:
            best_cycle_id = cycle_id
            best_cycle_score = cycle_score
            best_val = final_val
            best_model = copy.deepcopy(grouped_model)
            best_group_ids = picked_group_ids.copy()
            best_cycle_beats_anchor = final_val < anchor_val - cycle_improve_tol
            no_improve = 0
            log_fn(
                f"[Clean] New best grouped candidate at cycle {cycle_id}: "
                f"cycle_score={best_cycle_score:.6f}, val_mse={best_val:.6f}"
            )
        else:
            no_improve += 1
            log_fn(f"[Clean] No cycle-level improvement at cycle {cycle_id}; no_improve={no_improve}")

        _copy_grouped_solution_into_full_model(working_model, grouped_model, picked_group_ids)

        if warm_start_from_Zstar:
            Z_init = Z_star.detach().clone()
        else:
            Z_init = Z_final.detach().clone()
        U_init = None if reset_U_each_cycle else U_final.detach().clone()

        prev_Z_star = Z_star.detach().clone()
        prev_group_ids = picked_group_ids.copy()

        cycle_record = {
            "cycle": cycle_id,
            "all_Z": all_Z,
            "admm_metrics": metrics,
            "refit_results": refit_results,
            "picked_lambda_index": best_i,
            "picked_lambda_value": float(lambda_values[best_i]),
            "picked_n_groups": int(best_refit["n_groups"]),
            "picked_refit_val_mse": final_val,
            "picked_group_ids": picked_group_ids.copy(),
            "grouped_model": grouped_model,
            "cycle_start_val_mse": cycle_start_val,
            "cycle_val_mse": final_val,
            "anchor_gap": anchor_gap,
            "local_gap": local_gap,
            "cycle_score": cycle_score,
            "stability_to_prev": stability_to_prev,
            "picked_groups": copy.deepcopy(best_refit["groups"]),
            "cycle_start_full_model": copy.deepcopy(cycle_start_model),
        }
        all_cycles.append(cycle_record)

        stable_now = stability_to_prev is not None and stability_to_prev <= stability_tol
        if (not disable_cycle_early_stop) and (no_improve >= cycle_patience):
            if cycle_metric == "val":
                log_fn(f"[Clean] Early stop after cycle {cycle_id}: no improvement for {no_improve} cycles.")
                break
            if cycle_metric == "val_plus_stability" and stable_now:
                log_fn(
                    f"[Clean] Early stop after cycle {cycle_id}: "
                    f"no improvement for {no_improve} cycles and stable structure."
                )
                break

        if save_cycle_summary and outdir is not None:
            summary_path = os.path.join(outdir, f"cycle_summary_{cycle_id}.txt")
            with open(summary_path, "w") as f:
                f.write(f"cycle={cycle_id}\n")
                f.write(f"cycle_start_val_mse={cycle_start_val:.10f}\n")
                f.write(f"picked_lambda_index={best_i}\n")
                f.write(f"picked_lambda_value={float(lambda_values[best_i])}\n")
                f.write(f"picked_n_groups={int(best_refit['n_groups'])}\n")
                f.write(f"picked_refit_val_mse={final_val:.10f}\n")
                f.write(f"anchor_gap={anchor_gap:.10f}\n")
                f.write(f"local_gap={local_gap:.10f}\n")
                f.write(f"cycle_score={cycle_score:.10f}\n")
                f.write(f"stability_to_prev={stability_to_prev}\n")

    prediction_best_rec = _pick_cycle_by_prediction(all_cycles)
    structure_improve_tol = float(getattr(args, "structure_improve_tol", 0.0))
    structure_mse_rel_tol = float(getattr(args, "structure_mse_rel_tol", 0.05))
    structure_mse_abs_tol = float(getattr(args, "structure_mse_abs_tol", 0.0))
    structure_best_rec = _pick_cycle_by_structure(
        all_cycles,
        anchor_val=anchor_val,
        prediction_best_val=(
            float(prediction_best_rec["cycle_val_mse"])
            if prediction_best_rec is not None else None
        ),
        improve_tol=structure_improve_tol,
        mse_rel_tol=structure_mse_rel_tol,
        mse_abs_tol=structure_mse_abs_tol,
    )

    if prediction_best_rec is not None:
        log_fn(
            f"[Clean] Prediction-best cycle: cycle={int(prediction_best_rec['cycle'])}, "
            f"lambda_idx={int(prediction_best_rec['picked_lambda_index'])}, "
            f"n_groups={int(prediction_best_rec['picked_n_groups'])}, "
            f"val_mse={float(prediction_best_rec['cycle_val_mse']):.6f}"
        )
    if structure_best_rec is not None:
        log_fn(
            f"[Clean] Structure-best cycle: cycle={int(structure_best_rec['cycle'])}, "
            f"lambda_idx={int(structure_best_rec['picked_lambda_index'])}, "
            f"n_groups={int(structure_best_rec['picked_n_groups'])}, "
            f"val_mse={float(structure_best_rec['cycle_val_mse']):.6f} "
            f"(simplest grouped candidate near prediction-best)"
        )

    _copy_grouped_solution_into_full_model(model, best_model, best_group_ids)
    if best_cycle_id >= 1:
        log_fn(
            f"[Clean] Restored best grouped solution: best_cycle={best_cycle_id}, "
            f"best_cycle_score={best_cycle_score:.6f}, best_val_mse={best_val:.6f}, "
            f"beats_anchor={best_cycle_beats_anchor}"
        )
    else:
        log_fn("[Clean] Warning: no grouped cycle was recorded.")

    best_alt_cycle_id = None
    best_alt_cycle_score = None
    best_alt_cycle_val_mse = None
    best_alt_picked_lambda_index = None
    if prediction_best_rec is not None:
        best_alt_cycle_id = int(prediction_best_rec["cycle"])
        best_alt_cycle_score = float(prediction_best_rec["cycle_score"])
        best_alt_cycle_val_mse = float(prediction_best_rec["cycle_val_mse"])
        best_alt_picked_lambda_index = prediction_best_rec.get("picked_lambda_index", None)

    if outdir is not None and len(all_cycles) > 0:
        plot_pretrain_and_cycles(pretrain_val, all_cycles, outdir)

    oracle_true_group = _run_oracle_true_group_refit(
        anchor_model=anchor_model,
        true_group_ids=true_group_ids,
        model_kwargs=model_kwargs,
        target_scale=target_scale,
        X_train=X_train,
        A_train=A_train,
        y_train=y_train,
        X_val=X_val,
        A_val=A_val,
        y_val=y_val,
        X_test=X_test,
        A_test=A_test,
        y_test=y_test,
        args=args,
        log_fn=log_fn,
    )
    log_fn(
        "[Clean] Oracle true-group refit: "
        f"n_groups={int(oracle_true_group['n_groups'])}, "
        f"val_mse={float(oracle_true_group['val_mse']):.6f}, "
        f"test_mse={float(oracle_true_group['test_mse']):.6f}"
    )

    two_group_polish = None
    log_fn("[Clean] Two-group polish disabled for this value-first no-polish workflow.")

    best_picked_lambda_index = None
    best_picked_lambda_value = None
    best_picked_n_groups = None
    best_picked_refit_val_mse = None
    if best_cycle_id >= 1:
        rec = all_cycles[best_cycle_id - 1]
        best_picked_lambda_index = rec["picked_lambda_index"]
        best_picked_lambda_value = rec["picked_lambda_value"]
        best_picked_n_groups = rec["picked_n_groups"]
        best_picked_refit_val_mse = rec["picked_refit_val_mse"]

    return {
        "cycles": all_cycles,
        "best_cycle_id": best_cycle_id,
        "best_cycle_score": best_cycle_score,
        "best_val_mse": best_val,
        "best_picked_lambda_index": best_picked_lambda_index,
        "best_picked_lambda_value": best_picked_lambda_value,
        "best_picked_n_groups": best_picked_n_groups,
        "best_picked_refit_val_mse": best_picked_refit_val_mse,
        "best_model": best_model,
        "best_group_ids": best_group_ids,
        "pretrain_val_mse": pretrain_val,
        "anchor_val_mse": anchor_val,
        "best_cycle_beats_anchor": best_cycle_beats_anchor,
        "best_alt_cycle_id": best_alt_cycle_id,
        "best_alt_cycle_score": best_alt_cycle_score,
        "best_alt_cycle_val_mse": best_alt_cycle_val_mse,
        "best_alt_picked_lambda_index": best_alt_picked_lambda_index,
        "prediction_best_cycle_record": prediction_best_rec,
        "structure_best_cycle_record": structure_best_rec,
        "oracle_true_group_refit": oracle_true_group,
        "two_group_polish": two_group_polish,
    }
