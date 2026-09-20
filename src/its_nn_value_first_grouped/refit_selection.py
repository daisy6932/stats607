# refit_selection.py

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray


def _to_numpy(x: Any) -> np.ndarray:
    """
    Convert input to numpy array without forcing a copy when possible.
    """
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _safe_float(x: Any) -> float:
    """
    Convert scalar-like object to float.
    Handles python float, numpy scalar, torch scalar, etc.
    """
    try:
        return float(x)
    except Exception:
        if hasattr(x, "item"):
            return float(x.item())
        raise


def _row_close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """
    Compare two 1D vectors using max absolute deviation.
    """
    return np.max(np.abs(a - b)) <= tol


def count_groups_from_group_ids(group_ids: np.ndarray) -> int:
    """
    Return number of unique groups from group id vector.
    """
    group_ids = _to_numpy(group_ids).astype(int)
    if group_ids.size == 0:
        return 0
    return int(group_ids.max()) + 1


def partition_signature(group_ids: np.ndarray) -> Tuple[int, ...]:
    """
    Signature for deduplicating partitions.
    """
    group_ids = _to_numpy(group_ids).astype(int)
    return tuple(group_ids.tolist())


def groups_from_group_ids(group_ids: np.ndarray) -> List[List[int]]:
    """
    Convert group id vector to list of groups.
    Example:
        group_ids = [0,0,1,1,1] -> [[0,1],[2,3,4]]
    """
    group_ids = _to_numpy(group_ids).astype(int)
    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must be 1D, got shape={group_ids.shape}")

    n_groups = count_groups_from_group_ids(group_ids)
    groups: List[List[int]] = []
    for g in range(n_groups):
        members = np.where(group_ids == g)[0].tolist()
        groups.append(members)
    return groups


def extract_partition_from_Z(
    Z: np.ndarray,
    tol: float = 1e-6,
    round_decimals: int = 6,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Extract partition from fused treatment representation Z.

    Parameters
    ----------
    Z : np.ndarray, shape (K, r)
        Row k corresponds to treatment k.
    tol : float
        Max absolute difference threshold for saying two rows are same group.
    round_decimals : int
        Optional row rounding before grouping, helps stabilize floating comparisons.

    Returns
    -------
    groups : list[list[int]]
        Group membership list.
    group_ids : np.ndarray, shape (K,)
        group_ids[k] = group index for treatment k.
    """
    Z = _to_numpy(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")

    Zr = np.round(Z, round_decimals)
    K = Zr.shape[0]

    unique_rows: List[np.ndarray] = []
    group_ids = np.full(K, -1, dtype=int)

    for k in range(K):
        row = Zr[k]
        assigned = False
        for g, u in enumerate(unique_rows):
            if _row_close(row, u, tol=tol):
                group_ids[k] = g
                assigned = True
                break
        if not assigned:
            unique_rows.append(row.copy())
            group_ids[k] = len(unique_rows) - 1

    groups = groups_from_group_ids(group_ids)
    return groups, group_ids


def map_actions_to_groups(A, group_ids):
    """
    Map original treatment/action ids to grouped action ids.

    If A is a torch tensor, return a torch tensor on the same device.
    If A is a numpy array / list, return a numpy array.
    """
    import numpy as np

    # torch input -> torch output
    try:
        import torch
        is_torch = torch.is_tensor(A)
    except Exception:
        torch = None
        is_torch = False

    group_ids_np = np.asarray(group_ids).astype(int)

    if is_torch:
        A_np = A.detach().cpu().numpy().astype(int)

        if A_np.ndim != 1:
            raise ValueError(f"A must be 1D, got shape={A_np.shape}")

        if A_np.size > 0:
            if A_np.min() < 0 or A_np.max() >= len(group_ids_np):
                raise ValueError(
                    f"A contains invalid treatment ids. "
                    f"Allowed range is [0, {len(group_ids_np)-1}], got [{A_np.min()}, {A_np.max()}]."
                )

        out_np = group_ids_np[A_np]
        return torch.as_tensor(out_np, device=A.device, dtype=A.dtype)

    # numpy / list input -> numpy output
    A_np = np.asarray(A).astype(int)

    if A_np.ndim != 1:
        raise ValueError(f"A must be 1D, got shape={A_np.shape}")

    if A_np.size > 0:
        if A_np.min() < 0 or A_np.max() >= len(group_ids_np):
            raise ValueError(
                f"A contains invalid treatment ids. "
                f"Allowed range is [0, {len(group_ids_np)-1}], got [{A_np.min()}, {A_np.max()}]."
            )

    return group_ids_np[A_np]


def remap_group_ids_sequential(group_ids: np.ndarray) -> np.ndarray:
    """
    Ensure group ids are 0,1,...,G-1 in appearance order.

    Example:
        [2,2,5,5,9] -> [0,0,1,1,2]
    """
    group_ids = _to_numpy(group_ids).astype(int)
    mapping: Dict[int, int] = {}
    new_ids = np.empty_like(group_ids)

    nxt = 0
    for i, g in enumerate(group_ids):
        if g not in mapping:
            mapping[g] = nxt
            nxt += 1
        new_ids[i] = mapping[g]
    return new_ids


def _copy_matching_module(dst: Any, src: Any, name: str) -> None:
    """
    Copy one submodule state when both models expose it.
    """
    dst_mod = getattr(dst, name, None)
    src_mod = getattr(src, name, None)
    if dst_mod is None or src_mod is None:
        return
    dst_mod.load_state_dict(src_mod.state_dict())


def initialize_grouped_model_from_source(
    target_model: Any,
    source_model: Any,
    group_ids: np.ndarray,
) -> None:
    """
    Warm-start a grouped model from a full treatment model.

    Backbone and baseline are copied directly.
    Group-level gamma rows are initialized by averaging source gamma rows
    within each selected group.
    """
    import torch

    group_ids = remap_group_ids_sequential(group_ids)
    n_groups = count_groups_from_group_ids(group_ids)

    with torch.no_grad():
        _copy_matching_module(target_model, source_model, "fc1")
        _copy_matching_module(target_model, source_model, "fc2")
        _copy_matching_module(target_model, source_model, "m_head")

        src_gamma = source_model.gamma.weight.detach().to(
            device=target_model.gamma.weight.device,
            dtype=target_model.gamma.weight.dtype,
        )
        tgt_gamma = target_model.gamma.weight

        if tgt_gamma.shape[0] != n_groups:
            raise ValueError(
                f"target grouped gamma has {tgt_gamma.shape[0]} rows but "
                f"group_ids imply {n_groups} groups"
            )

        for g in range(n_groups):
            members = np.where(group_ids == g)[0]
            if len(members) == 0:
                continue
            idx = torch.as_tensor(
                members.tolist(),
                device=src_gamma.device,
                dtype=torch.long,
            )
            tgt_gamma[g].copy_(src_gamma.index_select(0, idx).mean(dim=0))


def run_grouped_refit(
    *,
    model_class: Callable[..., Any],
    train_fn: Callable[..., Any],
    eval_fn: Callable[..., Any],
    X_train: Any,
    A_train: Any,
    y_train: Any,
    X_val: Any,
    A_val: Any,
    y_val: Any,
    group_ids: np.ndarray,
    model_kwargs: Optional[Dict[str, Any]] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    model_n_actions_key: str = "n_actions",
    init_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Refit a model under a fixed partition.

    Parameters
    ----------
    model_class : callable
        Your model constructor, e.g. Net
    train_fn : callable
        Your supervised training function
        Expected use:
            model_out = train_fn(model, X_train, A_train_g, y_train, X_val, A_val_g, y_val, **train_kwargs)
        or
            model, hist = train_fn(...)
    eval_fn : callable
        Your evaluation function returning validation MSE or scalar loss
        Expected use:
            eval_fn(model, X_val, A_val_g, y_val)
    model_n_actions_key : str
        Constructor kwarg name for number of actions/groups. Usually 'n_actions'.

    Returns
    -------
    dict with keys:
        model, val_mse, n_groups, group_ids, train_output
    """
    model_kwargs = copy.deepcopy(model_kwargs or {})
    train_kwargs = copy.deepcopy(train_kwargs or {})

    group_ids = remap_group_ids_sequential(group_ids)
    n_groups = count_groups_from_group_ids(group_ids)

    A_train_g = map_actions_to_groups(A_train, group_ids)
    A_val_g = map_actions_to_groups(A_val, group_ids)

    model_kwargs[model_n_actions_key] = n_groups
    model = model_class(**model_kwargs)
    if init_model is not None:
        initialize_grouped_model_from_source(model, init_model, group_ids)

    train_output = train_fn(
        model,
        X_train, A_train_g, y_train,
        X_val, A_val_g, y_val,
        **train_kwargs,
    )

    # Default: training happens in-place
    trained_model = model

    # Optional compatibility:
    # if train_fn explicitly returns a model, use it
    try:
        import torch.nn as nn
        if isinstance(train_output, nn.Module):
            trained_model = train_output
        elif isinstance(train_output, tuple) and len(train_output) > 0:
            if isinstance(train_output[0], nn.Module):
                trained_model = train_output[0]
    except Exception:
        pass

    val_mse = eval_fn(trained_model, X_val, A_val_g, y_val)

    return {
        "model": trained_model,
        "val_mse": _safe_float(val_mse),
        "n_groups": n_groups,
        "group_ids": group_ids.copy(),
        "train_output": train_output,
    }


def eval_grouped_model_on_original_actions(
    model: Any,
    eval_fn: Callable[..., Any],
    X: Any,
    A: Any,
    y: Any,
    group_ids: np.ndarray,
) -> float:
    """
    Evaluate a grouped model on original treatment ids by remapping actions first.
    """
    A_grouped = map_actions_to_groups(A, group_ids)
    return _safe_float(eval_fn(model, X, A_grouped, y))


def evaluate_lambda_path_with_refit(
    *,
    Z_path: np.ndarray,
    lambda_values: Sequence[float],
    model_class: Callable[..., Any],
    train_fn: Callable[..., Any],
    eval_fn: Callable[..., Any],
    X_train: Any,
    A_train: Any,
    y_train: Any,
    X_val: Any,
    A_val: Any,
    y_val: Any,
    model_kwargs: Optional[Dict[str, Any]] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    init_model: Optional[Any] = None,
    partition_tol: float = 1e-6,
    round_decimals: int = 6,
    dedup_partitions: bool = True,
    model_n_actions_key: str = "n_actions",
    baseline_mse: Optional[float] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Evaluate every lambda on the path via grouped refit.

    Parameters
    ----------
    Z_path : np.ndarray, shape (L, K, r)
        Path of fused representations.
    lambda_values : sequence length L
    dedup_partitions : bool
        If True, identical partitions are refit only once and reused.

    Returns
    -------
    results : list[dict]
        Each dict contains:
            lambda_idx
            lambda
            groups
            group_ids
            n_groups
            refit_val_mse
            refit_model
            reused_from_idx (optional)
    """
    Z_path = _to_numpy(Z_path)
    if Z_path.ndim != 3:
        raise ValueError(f"Z_path must be 3D (L, K, r), got shape={Z_path.shape}")

    L = Z_path.shape[0]
    if len(lambda_values) != L:
        raise ValueError(
            f"lambda_values length {len(lambda_values)} does not match Z_path length {L}"
        )

    results: List[Dict[str, Any]] = []
    cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    for i in range(L):
        lam = _safe_float(lambda_values[i])
        Z = Z_path[i]

        groups, group_ids = extract_partition_from_Z(
            Z,
            tol=partition_tol,
            round_decimals=round_decimals,
        )
        group_ids = remap_group_ids_sequential(group_ids)
        sig = partition_signature(group_ids)

        reused_from_idx: Optional[int] = None

        if dedup_partitions and sig in cache:
            cached = cache[sig]
            refit_model = cached["refit_model"]
            refit_val_mse = cached["refit_val_mse"]
            n_groups = cached["n_groups"]
            reused_from_idx = cached["lambda_idx"]
        else:
            out = run_grouped_refit(
                model_class=model_class,
                train_fn=train_fn,
                eval_fn=eval_fn,
                X_train=X_train,
                A_train=A_train,
                y_train=y_train,
                X_val=X_val,
                A_val=A_val,
                y_val=y_val,
                group_ids=group_ids,
                model_kwargs=model_kwargs,
                train_kwargs=train_kwargs,
                model_n_actions_key=model_n_actions_key,
                init_model=init_model,
            )
            refit_model = out["model"]
            refit_val_mse = out["val_mse"]
            n_groups = out["n_groups"]

            if dedup_partitions:
                cache[sig] = {
                    "lambda_idx": i,
                    "refit_model": refit_model,
                    "refit_val_mse": refit_val_mse,
                    "n_groups": n_groups,
                }

        row = {
            "lambda_idx": i,
            "lambda": lam,
            "groups": groups,
            "group_ids": group_ids.copy(),
            "n_groups": int(n_groups),
            "refit_val_mse": float(refit_val_mse),
            "delta_vs_baseline": (
                float(refit_val_mse) - float(baseline_mse)
                if baseline_mse is not None else None
            ),
            "refit_model": refit_model,
            "reused_from_idx": reused_from_idx,
        }
        results.append(row)

        if verbose:
            reuse_msg = f" [reused from {reused_from_idx}]" if reused_from_idx is not None else ""
            print(
                f"[refit] idx={i:03d}  lambda={lam:.6g}  "
                f"groups={n_groups}  val_mse={refit_val_mse:.6f}{reuse_msg}"
            )

    return results


def pick_lambda_by_refit_min(
    refit_results: Sequence[Dict[str, Any]],
) -> int:
    """
    Pick lambda with smallest refit_val_mse.
    Tie-break:
        1) smaller n_groups
        2) smaller lambda_idx
    """
    if len(refit_results) == 0:
        raise ValueError("refit_results is empty")

    best_idx = min(
        range(len(refit_results)),
        key=lambda i: (
            refit_results[i]["refit_val_mse"],
            refit_results[i]["n_groups"],
            refit_results[i]["lambda_idx"],
        ),
    )
    return int(best_idx)


def pick_lambda_by_refit_eps_simplest(
    refit_results: Sequence[Dict[str, Any]],
    eps: float = 0.05,
) -> int:
    """
    Pick simplest partition among lambdas whose refit_val_mse is within
    (1 + eps) of the minimum.

    Rule:
        1) m* = min refit_val_mse
        2) candidate if mse <= m*(1+eps)
        3) among candidates choose smallest n_groups
        4) tie-break by smaller mse, then smaller lambda_idx
    """
    if len(refit_results) == 0:
        raise ValueError("refit_results is empty")
    if eps < 0:
        raise ValueError(f"eps must be nonnegative, got {eps}")

    mses = np.array([float(r["refit_val_mse"]) for r in refit_results], dtype=float)
    best_mse = float(mses.min())
    cutoff = best_mse * (1.0 + eps)

    candidates = [i for i, r in enumerate(refit_results) if float(r["refit_val_mse"]) <= cutoff]
    best_idx = min(
        candidates,
        key=lambda i: (
            refit_results[i]["n_groups"],
            refit_results[i]["refit_val_mse"],
            refit_results[i]["lambda_idx"],
        ),
    )
    return int(best_idx)


def pick_lambda_by_refit_improve_eps_simplest(
    refit_results: Sequence[Dict[str, Any]],
    baseline_mse: float,
    eps: float = 0.05,
    improve_tol: float = 0.0,
) -> int:
    """
    Prefer partitions that do not underperform the baseline model.

    Rule
    ----
    1) keep lambdas with refit_val_mse <= baseline_mse + improve_tol
    2) if such candidates exist, among them keep those within (1 + eps) of the
       best candidate mse, then choose the simplest partition
    3) otherwise fall back to global minimum refit_val_mse
    """
    if len(refit_results) == 0:
        raise ValueError("refit_results is empty")

    improving = [
        i for i, r in enumerate(refit_results)
        if float(r["refit_val_mse"]) <= float(baseline_mse) + float(improve_tol)
    ]
    if not improving:
        return pick_lambda_by_refit_min(refit_results)

    best_mse = min(float(refit_results[i]["refit_val_mse"]) for i in improving)
    cutoff = best_mse * (1.0 + float(eps))
    candidates = [
        i for i in improving
        if float(refit_results[i]["refit_val_mse"]) <= cutoff
    ]
    best_idx = min(
        candidates,
        key=lambda i: (
            refit_results[i]["n_groups"],
            refit_results[i]["refit_val_mse"],
            refit_results[i]["lambda_idx"],
        ),
    )
    return int(best_idx)


def pick_lambda_by_refit_improve_eps_prefer_k(
    refit_results: Sequence[Dict[str, Any]],
    baseline_mse: float,
    eps: float = 0.05,
    improve_tol: float = 0.0,
    preferred_n_groups: int = 2,
) -> int:
    """
    Legacy compatibility rule.

    Prefer partitions whose group count is closest to ``preferred_n_groups``
    among candidates that remain competitive in validation MSE.

    Rule
    ----
    1) keep lambdas with refit_val_mse <= baseline_mse + improve_tol
    2) among them keep those within (1 + eps) of the best improving mse
    3) choose the candidate with n_groups closest to preferred_n_groups
    4) tie-break by smaller n_groups, then smaller mse, then lambda_idx
    5) if no improving candidates exist, fall back to global minimum refit_val_mse
    """
    if len(refit_results) == 0:
        raise ValueError("refit_results is empty")

    improving = [
        i for i, r in enumerate(refit_results)
        if float(r["refit_val_mse"]) <= float(baseline_mse) + float(improve_tol)
    ]
    if not improving:
        return pick_lambda_by_refit_min(refit_results)

    best_mse = min(float(refit_results[i]["refit_val_mse"]) for i in improving)
    cutoff = best_mse * (1.0 + float(eps))
    candidates = [
        i for i in improving
        if float(refit_results[i]["refit_val_mse"]) <= cutoff
    ]
    best_idx = min(
        candidates,
        key=lambda i: (
            abs(int(refit_results[i]["n_groups"]) - int(preferred_n_groups)),
            int(refit_results[i]["n_groups"]),
            float(refit_results[i]["refit_val_mse"]),
            int(refit_results[i]["lambda_idx"]),
        ),
    )
    return int(best_idx)


def pick_lambda_by_refit_score(
    refit_results: Sequence[Dict[str, Any]],
    alpha: float = 0.0,
) -> int:
    """
    Pick lambda minimizing:
        score = refit_val_mse + alpha * n_groups

    Useful when you want an explicit complexity tradeoff.
    """
    if len(refit_results) == 0:
        raise ValueError("refit_results is empty")

    best_idx = min(
        range(len(refit_results)),
        key=lambda i: (
            refit_results[i]["refit_val_mse"] + alpha * refit_results[i]["n_groups"],
            refit_results[i]["refit_val_mse"],
            refit_results[i]["n_groups"],
            refit_results[i]["lambda_idx"],
        ),
    )
    return int(best_idx)


def pick_lambda_by_refit(
    refit_results: Sequence[Dict[str, Any]],
    rule: str = "eps_simplest",
    eps: float = 0.05,
    alpha: float = 0.0,
    baseline_mse: Optional[float] = None,
    improve_tol: float = 0.0,
    preferred_n_groups: int = 2,
) -> int:
    """
    Unified lambda picker.

    rule options
    ------------
    "min"
        smallest refit_val_mse
    "eps_simplest"
        within eps of best mse, choose simplest partition
    "score"
        minimize refit_val_mse + alpha * n_groups
    "improve_eps_simplest"
        among partitions no worse than baseline + improve_tol, choose the
        simplest one within eps of the best improving mse; else fall back to min
    "improve_eps_prefer_k"
        legacy compatibility rule; prefer the partition whose group count is
        closest to preferred_n_groups within eps of the best improving mse
    """
    rule = rule.lower().strip()

    if rule == "min":
        return pick_lambda_by_refit_min(refit_results)
    if rule == "eps_simplest":
        return pick_lambda_by_refit_eps_simplest(refit_results, eps=eps)
    if rule == "score":
        return pick_lambda_by_refit_score(refit_results, alpha=alpha)
    if rule == "improve_eps_simplest":
        if baseline_mse is None:
            raise ValueError("baseline_mse is required for improve_eps_simplest")
        return pick_lambda_by_refit_improve_eps_simplest(
            refit_results,
            baseline_mse=baseline_mse,
            eps=eps,
            improve_tol=improve_tol,
        )
    if rule == "improve_eps_prefer_k":
        if baseline_mse is None:
            raise ValueError("baseline_mse is required for improve_eps_prefer_k")
        return pick_lambda_by_refit_improve_eps_prefer_k(
            refit_results,
            baseline_mse=baseline_mse,
            eps=eps,
            improve_tol=improve_tol,
            preferred_n_groups=preferred_n_groups,
        )

    raise ValueError(f"Unknown refit selection rule: {rule}")


def summarize_refit_results(
    refit_results: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Produce lightweight serializable summary rows.
    """
    rows: List[Dict[str, Any]] = []
    for r in refit_results:
        rows.append({
            "lambda_idx": int(r["lambda_idx"]),
            "lambda": float(r["lambda"]),
            "n_groups": int(r["n_groups"]),
            "refit_val_mse": float(r["refit_val_mse"]),
            "groups": r["groups"],
            "reused_from_idx": r.get("reused_from_idx"),
        })
    return rows


def save_refit_metrics_csv(
    refit_results: Sequence[Dict[str, Any]],
    out_csv: str | Path,
) -> None:
    """
    Save compact refit metrics to CSV.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "lambda_idx",
        "lambda",
        "n_groups",
        "refit_val_mse",
        "delta_vs_baseline",
        "groups",
        "group_ids",
        "reused_from_idx",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in refit_results:
            writer.writerow({
                "lambda_idx": int(r["lambda_idx"]),
                "lambda": float(r["lambda"]),
                "n_groups": int(r["n_groups"]),
                "refit_val_mse": float(r["refit_val_mse"]),
                "delta_vs_baseline": r.get("delta_vs_baseline"),
                "groups": str(r["groups"]),
                "group_ids": str(_to_numpy(r["group_ids"]).astype(int).tolist()),
                "reused_from_idx": r.get("reused_from_idx"),
            })


def evaluate_test_mse_for_selected_refit(
    *,
    selected_refit_result: Dict[str, Any],
    eval_fn: Callable[..., Any],
    X_test: Any,
    A_test: Any,
    y_test: Any,
) -> float:
    """
    Evaluate selected refit model on test set.

    This uses the selected partition to map original actions to grouped actions.
    """
    group_ids = _to_numpy(selected_refit_result["group_ids"]).astype(int)
    model = selected_refit_result["refit_model"]

    A_test_g = map_actions_to_groups(A_test, group_ids)
    test_mse = eval_fn(model, X_test, A_test_g, y_test)
    return _safe_float(test_mse)
