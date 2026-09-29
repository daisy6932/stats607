# plotting.py

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import scipy.cluster.hierarchy as sch


# =========================================================
# basic saves
# =========================================================


def _shift_linkage_heights_nonnegative(Z_link, extra_heights=None):
    """
    SciPy linkage matrices cannot contain negative distances.

    Some of our path-summary plots use log(lambda * scale) as the visual height,
    which can be negative for small lambdas. For plotting only, shift the entire
    height axis upward by a constant so the minimum plotted height is 0 while
    preserving merge order and relative spacing.
    """
    Z_link = np.asarray(Z_link, dtype=float).copy()
    extras = [] if extra_heights is None else [float(h) for h in extra_heights]

    if Z_link.size == 0:
        return Z_link, extras, 0.0

    min_h = float(np.min(Z_link[:, 2]))
    if extras:
        min_h = min(min_h, min(extras))

    if min_h < 0.0:
        shift = -min_h
        Z_link[:, 2] += shift
        extras = [h + shift for h in extras]
        return Z_link, extras, shift

    return Z_link, extras, 0.0

def save_path_metrics_csv(metrics, path):
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in metrics:
            w.writerow(m)


def save_true_group_heatmap(groups, n_treatments, save_path, title="True Group Structure"):
    """
    groups: e.g. [[0,1,2,3,4],[5,6,7,8,9]]
    Save a KxK matrix where entry (i,j)=1 if i,j are in same true group, else 0.
    """
    M = np.zeros((n_treatments, n_treatments), dtype=float)

    for grp in groups:
        for i in grp:
            for j in grp:
                M[i, j] = 1.0

    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Treatment Index")
    plt.ylabel("Treatment Index")
    plt.xticks(range(n_treatments))
    plt.yticks(range(n_treatments))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def save_partition_heatmap_from_group_ids(
    group_ids,
    save_path,
    title="Selected Partition Structure",
    xlabel="Treatment",
    ylabel="Treatment",
):
    """
    Save a same-group matrix directly from group ids.
    """
    group_ids = np.asarray(group_ids).astype(int)
    K = len(group_ids)
    M = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            M[i, j] = 1.0 if group_ids[i] == group_ids[j] else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Same Selected Group", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(K), [str(i + 1) for i in range(K)])
    ax.set_yticks(range(K), [str(i + 1) for i in range(K)])
    ax.set_facecolor("#f7f7f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.8)
    ax.text(
        0.99,
        0.02,
        "Final selected partition",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_selected_partition_dendrogram_from_group_ids(
    group_ids,
    save_path,
    title="Selected Refit Partition Dendrogram",
    xlabel="Treatment",
    ylabel="Partition Height",
):
    """
    Visualize the final selected partition itself as a dendrogram.

    This is intentionally *not* a path summary. Heights are synthetic and only
    encode the final grouping structure:
    - small heights merge items within the same selected group
    - larger heights merge different groups
    """
    group_ids = np.asarray(group_ids).astype(int)
    K = len(group_ids)
    if K == 0:
        return None

    # preserve first-appearance order so labels match the partition heatmap
    ordered_groups = []
    seen = set()
    for gid in group_ids:
        gid = int(gid)
        if gid not in seen:
            seen.add(gid)
            ordered_groups.append(gid)

    members_by_gid = {
        gid: [i for i, g in enumerate(group_ids) if int(g) == gid]
        for gid in ordered_groups
    }

    linkage = []
    next_id = K
    cluster_sizes = {i: 1 for i in range(K)}
    group_roots = []

    # Merge within each selected group at low heights.
    for gidx, gid in enumerate(ordered_groups):
        members = members_by_gid[gid]
        if len(members) == 1:
            group_roots.append(members[0])
            continue

        current = members[0]
        inner_base = 0.20 + 0.05 * (gidx % 4)
        inner_step = 0.08
        for j, nxt in enumerate(members[1:], start=1):
            new_id = next_id
            height = inner_base + inner_step * j
            size = cluster_sizes[current] + cluster_sizes[nxt]
            linkage.append([current, nxt, height, size])
            cluster_sizes[new_id] = size
            current = new_id
            next_id += 1
        group_roots.append(current)

    # Merge the selected groups at larger heights so the final partition is clear.
    if len(group_roots) > 1:
        current = group_roots[0]
        outer_base = 1.40
        outer_step = 0.35
        for j, nxt in enumerate(group_roots[1:], start=1):
            new_id = next_id
            height = outer_base + outer_step * j
            size = cluster_sizes[current] + cluster_sizes[nxt]
            linkage.append([current, nxt, height, size])
            cluster_sizes[new_id] = size
            current = new_id
            next_id += 1

    if not linkage:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.text(0.5, 0.5, "Single selected group", ha="center", va="center", fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_facecolor("#f7f7f7")
        for spine in ax.spines.values():
            spine.set_edgecolor("#666666")
            spine.set_linewidth(0.8)
        ax.text(
            0.99,
            0.02,
            "Final selected partition only",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return save_path

    Z_link = np.array(linkage, dtype=float)
    labels = [str(i + 1) for i in range(K)]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sch.dendrogram(
        Z_link,
        labels=labels,
        leaf_rotation=0.0,
        leaf_font_size=11,
        color_threshold=1.0,
        above_threshold_color="#4c4c4c",
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_facecolor("#f7f7f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.8)
    ax.text(
        0.99,
        0.02,
        "Final selected partition only\nHeights are schematic, not lambda values",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =========================================================
# clustering helpers
# =========================================================

def _cluster_ids_by_round(Z: np.ndarray, round_decimals: int) -> np.ndarray:
    Zr = np.round(Z, round_decimals)
    _, cid = np.unique(Zr, axis=0, return_inverse=True)
    return cid


def _cluster_ids_by_eps(Z: np.ndarray, snap_eps: float) -> np.ndarray:
    K = Z.shape[0]
    cid = -np.ones(K, dtype=int)
    cur = 0
    for i in range(K):
        if cid[i] != -1:
            continue
        cid[i] = cur
        for j in range(i + 1, K):
            if cid[j] != -1:
                continue
            if np.max(np.abs(Z[i] - Z[j])) <= snap_eps:
                cid[j] = cur
        cur += 1
    return cid


# =========================================================
# selected structure plots
# =========================================================

def save_selected_Z_heatmap(Z, save_path, title="Selected Z* pairwise distance"):
    """
    Z: (K, d) selected fused parameter matrix
    Save pairwise row distance heatmap.
    """
    Z = np.asarray(Z)
    K = Z.shape[0]
    D = np.zeros((K, K), dtype=float)

    for i in range(K):
        for j in range(K):
            D[i, j] = np.linalg.norm(Z[i] - Z[j])

    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Treatment Index")
    plt.ylabel("Treatment Index")
    plt.xticks(range(K))
    plt.yticks(range(K))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def save_selected_cluster_heatmap(Z, save_path, round_decimals=3, title="Selected Cluster Structure"):
    """
    Convert selected Z into cluster ids and save same-cluster matrix.
    """
    Z = np.asarray(Z)
    Zr = np.round(Z, round_decimals)
    _, cid = np.unique(Zr, axis=0, return_inverse=True)

    K = len(cid)
    M = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            M[i, j] = 1.0 if cid[i] == cid[j] else 0.0

    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Treatment Index")
    plt.ylabel("Treatment Index")
    plt.xticks(range(K))
    plt.yticks(range(K))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# =========================================================
# cycle summary
# =========================================================

def plot_cycle_summary(cycle_records, save_path):
    """
    cycle_records: list of dicts, each containing:
      - cycle
      - cycle_val_mse
      - cycle_score
      - stability_to_prev
    """
    if not cycle_records:
        return None

    cycles = [r["cycle"] for r in cycle_records]
    val_mse = [r.get("cycle_val_mse", np.nan) for r in cycle_records]
    cycle_score = [r.get("cycle_score", np.nan) for r in cycle_records]
    stability = [r.get("stability_to_prev", np.nan) for r in cycle_records]

    plt.figure(figsize=(8, 5))
    plt.plot(cycles, val_mse, marker="o", label="Validation MSE")
    plt.plot(cycles, cycle_score, marker="s", label="Cycle Score")
    plt.plot(cycles, stability, marker="^", label="Stability to Prev")
    plt.xlabel("Cycle")
    plt.ylabel("Value")
    plt.title("Cycle-level Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# =========================================================
# old lambda curve (kept for compatibility)
# =========================================================

def plot_lambda_selection_curve(metrics, picked_lambda_index, save_path):
    """
    Old path-metric plot.
    metrics: list of dicts from one cycle
    """
    if not metrics:
        return None

    lambdas = [m["lambda"] for m in metrics]
    reg_loss = [m.get("reg_loss", np.nan) for m in metrics]
    val_mse = [m.get("val_mse", np.nan) for m in metrics]
    c3 = [m.get("clusters_3d", np.nan) for m in metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, reg_loss, marker="o", label="Reg Loss")
    if not all(np.isnan(val_mse)):
        plt.plot(lambdas, val_mse, marker="s", label="Val MSE")
    plt.plot(lambdas, c3, marker="^", label="Clusters_3d")

    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambdas):
        x = lambdas[picked_lambda_index]
        plt.axvline(x=x, linestyle="--", label=f"Picked λ={x:.4f}")

    plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel("Lambda")
    plt.ylabel("Metric")
    plt.title("Lambda Selection Curve (Original)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# =========================================================
# new refit-based lambda plots
# =========================================================

def plot_refit_lambda_selection_curve(
    lambda_values,
    refit_results,
    picked_lambda_index,
    save_path,
    old_metrics=None,
):
    """
    Main plot for the new workflow.

    Parameters
    ----------
    lambda_values : sequence
    refit_results : list[dict]
        output from evaluate_lambda_path_with_refit
    picked_lambda_index : int
    old_metrics : optional list[dict]
        original ADMM path metrics; if provided, original val_mse will be overlayed
    """
    if not refit_results:
        return None

    lambdas = [float(x) for x in lambda_values]
    refit_val_mse = [float(r["refit_val_mse"]) for r in refit_results]
    n_groups = [int(r["n_groups"]) for r in refit_results]

    old_val_mse = None
    if old_metrics is not None and len(old_metrics) == len(lambdas):
        vals = []
        has_any = False
        for m in old_metrics:
            v = m.get("val_mse", np.nan)
            if v is not None:
                has_any = True
            vals.append(np.nan if v is None else float(v))
        if has_any:
            old_val_mse = vals

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(lambdas, refit_val_mse, marker="o", label="Refit Val MSE")
    if old_val_mse is not None:
        ax1.plot(lambdas, old_val_mse, marker="s", label="Original Val MSE")

    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambdas):
        x = lambdas[picked_lambda_index]
        ax1.axvline(x=x, linestyle="--", label=f"Picked λ={x:.4g}")

    ax1.set_xscale("symlog", linthresh=1e-4)
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Validation MSE")
    ax1.set_title("Refit-based Lambda Selection")

    ax2 = ax1.twinx()
    ax2.plot(lambdas, n_groups, marker="^", label="# Groups")
    ax2.set_ylabel("Number of Groups")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_refit_mse_vs_lambda(
    lambda_values,
    refit_results,
    picked_lambda_index,
    save_path,
    old_metrics=None,
):
    """
    Simpler version: only plot MSE curves.
    """
    if not refit_results:
        return None

    lambdas = [float(x) for x in lambda_values]
    refit_val_mse = [float(r["refit_val_mse"]) for r in refit_results]

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, refit_val_mse, marker="o", label="Refit Val MSE")

    if old_metrics is not None and len(old_metrics) == len(lambdas):
        old_val_mse = [m.get("val_mse", np.nan) for m in old_metrics]
        if not all(np.isnan(old_val_mse)):
            plt.plot(lambdas, old_val_mse, marker="s", label="Original Val MSE")

    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambdas):
        x = lambdas[picked_lambda_index]
        plt.axvline(x=x, linestyle="--", label=f"Picked λ={x:.4g}")

    plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel("Lambda")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE vs Lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_groups_vs_lambda(
    lambda_values,
    refit_results,
    picked_lambda_index,
    save_path,
):
    """
    Plot number of groups vs lambda.
    """
    if not refit_results:
        return None

    lambdas = [float(x) for x in lambda_values]
    n_groups = [int(r["n_groups"]) for r in refit_results]

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, n_groups, marker="o", label="# Groups")

    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambdas):
        x = lambdas[picked_lambda_index]
        plt.axvline(x=x, linestyle="--", label=f"Picked λ={x:.4g}")

    plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel("Lambda")
    plt.ylabel("Number of Groups")
    plt.title("Group Count vs Lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_solution_path_dendrogram(
    lambda_values,
    all_Z_list,
    picked_lambda_index,
    save_path,
    round_decimals=2,
    snap_eps=None,
    labels=None,
    xlabel="Treatment",
    ylabel=r"$\log(\lambda \times 1000)$",
    title=None,
    lambda_scale=1000.0,
):
    """
    Make a publication-style solution-path summary dendrogram.

    Important:
    - This is a path summary built from the sequence of Z matrices.
    - The dashed line marks the selected lambda height on the path.
    - The final selected partition should be read from the partition plot / CSV,
      not inferred only from this dendrogram.
    """
    if not all_Z_list:
        return None

    L = len(all_Z_list)
    K = all_Z_list[0].shape[0]

    # ---------- clustering path ----------
    all_clusterings = []
    for Z in all_Z_list:
        Z = np.asarray(Z)
        if Z.shape[0] != K:
            raise ValueError("Inconsistent K across all_Z_list.")
        if snap_eps is None:
            cid = _cluster_ids_by_round(Z, round_decimals)
        else:
            cid = _cluster_ids_by_eps(Z, float(snap_eps))
        all_clusterings.append(cid)

    # ---------- build linkage ----------
    linkage_list = []
    cluster_members = {i: frozenset([i]) for i in range(K)}
    cluster_heights = {i: 0.0 for i in range(K)}
    next_id = K
    merges_found = False

    def _transform_height(lam):
        lam = float(lam)
        lam = max(lam, 1e-8)
        return np.log(lam * lambda_scale)

    for t in range(1, L):
        lam = float(lambda_values[t])
        h = _transform_height(lam)
        groups = all_clusterings[t]

        merger_map = {}
        active = list(cluster_members.keys())

        for cid in active:
            item = next(iter(cluster_members[cid]))
            new_gid = int(groups[item])
            merger_map.setdefault(new_gid, set()).add(cid)

        for _, to_merge in merger_map.items():
            if len(to_merge) <= 1:
                continue
            merges_found = True

            lst = sorted(list(to_merge), key=lambda x: cluster_heights[x])

            while len(lst) > 1:
                a = lst.pop(0)
                b = lst.pop(0)

                new_id = next_id
                new_h = h
                new_mem = cluster_members[a].union(cluster_members[b])

                linkage_list.append([a, b, new_h, len(new_mem)])

                del cluster_members[a]
                del cluster_members[b]
                cluster_members[new_id] = new_mem
                cluster_heights[new_id] = new_h
                next_id += 1

                lst.append(new_id)
                lst.sort(key=lambda x: cluster_heights[x])

    if (not linkage_list) or (not merges_found):
        plt.figure(figsize=(8, 4.8))
        ax = plt.gca()
        if title is not None:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.text(
            0.5, 0.5,
            "No merges found along lambda path",
            ha="center", va="center", fontsize=12
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambda_values):
            picked_lambda = float(lambda_values[picked_lambda_index])
            picked_lambda = max(picked_lambda, 1e-8)
            picked_h = np.log(picked_lambda * lambda_scale)
            ax.axhline(
                y=picked_h,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )

        ax.text(
            0.99,
            0.02,
            "Dashed line = selected lambda height\nRead final partition from the partition plot / CSV",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()
        return save_path

    # complete tree if needed
    if len(linkage_list) < K - 1:
        last_height = float(max(link[2] for link in linkage_list)) if linkage_list else 0.0
        eps = max(1e-8, 1e-6 * max(1.0, abs(last_height)))

        remaining = sorted(cluster_members.keys(), key=lambda x: cluster_heights[x])
        while len(remaining) > 1:
            a = remaining.pop(0)
            b = remaining.pop(0)

            new_id = next_id
            new_h = max(cluster_heights[a], cluster_heights[b], last_height) + eps
            new_mem = cluster_members[a].union(cluster_members[b])

            linkage_list.append([a, b, new_h, len(new_mem)])

            del cluster_members[a]
            del cluster_members[b]
            cluster_members[new_id] = new_mem
            cluster_heights[new_id] = new_h
            next_id += 1

            remaining.append(new_id)
            remaining.sort(key=lambda x: cluster_heights[x])

    Z_link = np.array(linkage_list, dtype=float)

    if Z_link.shape[0] != K - 1:
        raise ValueError(f"Invalid linkage matrix: got {Z_link.shape[0]} merges, expected {K-1}.")

    if labels is None:
        labels = [str(i + 1) for i in range(K)]  # Use labels 1..K for publication figures.
    else:
        if len(labels) != K:
            raise ValueError("labels length must equal K.")

    picked_h = None
    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambda_values):
        picked_lambda = float(lambda_values[picked_lambda_index])
        picked_h = _transform_height(picked_lambda)
    Z_link, shifted_heights, height_shift = _shift_linkage_heights_nonnegative(
        Z_link,
        [picked_h] if picked_h is not None else None,
    )
    if picked_h is not None:
        picked_h = shifted_heights[0]

    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()

    sch.dendrogram(
        Z_link,
        labels=labels,
        leaf_rotation=0.0,
        leaf_font_size=11,
        ax=ax,
        color_threshold=None,
    )

    if picked_h is not None:
        ax.axhline(
            y=picked_h,
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )

    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    ax.set_xlabel(xlabel, fontsize=12)
    plotted_ylabel = ylabel if height_shift <= 0 else f"Shifted {ylabel}"
    ax.set_ylabel(plotted_ylabel, fontsize=12)

    ax.text(
        0.99,
        0.02,
        "Path summary only\nDashed line = selected lambda height",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    return save_path


def plot_selected_cycle_path_with_group_bars(
    lambda_values,
    all_Z_list,
    picked_lambda_index,
    selected_group_ids,
    true_group_ids,
    save_path,
    round_decimals=2,
    snap_eps=None,
    labels=None,
    title=None,
    lambda_scale=1000.0,
):
    """
    Paper-style summary figure:
    - top: path-summary dendrogram for the selected cycle
    - bottom bars: true grouping vs selected grouping

    Important:
    - The dendrogram is still a path summary for the selected cycle.
    - The bottom bars make the final chosen grouping explicit, which avoids
      over-interpreting the tree alone.
    """
    if not all_Z_list:
        return None

    L = len(all_Z_list)
    K = all_Z_list[0].shape[0]
    selected_group_ids = np.asarray(selected_group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)

    if labels is None:
        labels = [str(i + 1) for i in range(K)]
    if len(labels) != K:
        raise ValueError("labels length must equal K")

    all_clusterings = []
    for Z in all_Z_list:
        Z = np.asarray(Z)
        if snap_eps is None:
            cid = _cluster_ids_by_round(Z, round_decimals)
        else:
            cid = _cluster_ids_by_eps(Z, float(snap_eps))
        all_clusterings.append(cid)

    linkage_list = []
    cluster_members = {i: frozenset([i]) for i in range(K)}
    cluster_heights = {i: 0.0 for i in range(K)}
    next_id = K

    def _transform_height(lam):
        lam = max(float(lam), 1e-8)
        return np.log(lam * lambda_scale)

    for t in range(1, L):
        h = _transform_height(float(lambda_values[t]))
        groups = all_clusterings[t]
        merger_map = {}
        active = list(cluster_members.keys())
        for cid in active:
            item = next(iter(cluster_members[cid]))
            gid = int(groups[item])
            merger_map.setdefault(gid, set()).add(cid)

        for to_merge in merger_map.values():
            if len(to_merge) <= 1:
                continue
            lst = sorted(list(to_merge), key=lambda x: cluster_heights[x])
            while len(lst) > 1:
                a = lst.pop(0)
                b = lst.pop(0)
                new_mem = cluster_members[a].union(cluster_members[b])
                linkage_list.append([a, b, h, len(new_mem)])
                del cluster_members[a]
                del cluster_members[b]
                cluster_members[next_id] = new_mem
                cluster_heights[next_id] = h
                lst.append(next_id)
                lst.sort(key=lambda x: cluster_heights[x])
                next_id += 1

    if len(linkage_list) < K - 1:
        last_height = float(max((link[2] for link in linkage_list), default=0.0))
        eps = max(1e-8, 1e-6 * max(1.0, abs(last_height)))
        remaining = sorted(cluster_members.keys(), key=lambda x: cluster_heights[x])
        while len(remaining) > 1:
            a = remaining.pop(0)
            b = remaining.pop(0)
            new_h = max(cluster_heights[a], cluster_heights[b], last_height) + eps
            new_mem = cluster_members[a].union(cluster_members[b])
            linkage_list.append([a, b, new_h, len(new_mem)])
            del cluster_members[a]
            del cluster_members[b]
            cluster_members[next_id] = new_mem
            cluster_heights[next_id] = new_h
            remaining.append(next_id)
            remaining.sort(key=lambda x: cluster_heights[x])
            next_id += 1

    if len(linkage_list) != K - 1:
        return None

    Z_link = np.array(linkage_list, dtype=float)
    picked_h = None
    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambda_values):
        picked_lambda = float(lambda_values[picked_lambda_index])
        picked_h = _transform_height(picked_lambda)
    Z_link, shifted_heights, height_shift = _shift_linkage_heights_nonnegative(
        Z_link,
        [picked_h] if picked_h is not None else None,
    )
    if picked_h is not None:
        picked_h = shifted_heights[0]

    fig = plt.figure(figsize=(9.2, 6.6))
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[5.2, 0.42, 0.42],
        hspace=0.12,
    )
    ax_tree = fig.add_subplot(gs[0])
    dendro = sch.dendrogram(
        Z_link,
        labels=labels,
        leaf_rotation=0.0,
        leaf_font_size=11,
        color_threshold=None,
        ax=ax_tree,
    )

    if picked_h is not None:
        ax_tree.axhline(
            y=picked_h,
            linestyle="--",
            linewidth=1.4,
            alpha=0.95,
            color="#c23b22",
        )
        ax_tree.text(
            0.01,
            0.97,
            f"Selected λ = {picked_lambda:.4g}",
            transform=ax_tree.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#d9d9d9"),
        )

    ax_tree.set_facecolor("#f7f7f7")
    for spine in ax_tree.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.8)
    tree_ylabel = r"$\log(\lambda \times 1000)$"
    if height_shift > 0:
        tree_ylabel = f"Shifted {tree_ylabel}"
    ax_tree.set_ylabel(tree_ylabel, fontsize=12)
    ax_tree.set_xlabel("")
    if title is not None:
        ax_tree.set_title(title, fontsize=14, fontweight="bold", pad=10)

    leaf_order = dendro["leaves"]
    true_reordered = true_group_ids[leaf_order][None, :]
    selected_reordered = selected_group_ids[leaf_order][None, :]
    n_colors = int(max(true_group_ids.max(), selected_group_ids.max()) + 1)
    cmap = ListedColormap(plt.cm.Set3(np.linspace(0, 1, max(n_colors, 3))))

    ax_true = fig.add_subplot(gs[1])
    ax_sel = fig.add_subplot(gs[2])
    ax_true.imshow(true_reordered, aspect="auto", cmap=cmap, vmin=0, vmax=max(n_colors - 1, 1))
    ax_sel.imshow(selected_reordered, aspect="auto", cmap=cmap, vmin=0, vmax=max(n_colors - 1, 1))

    for ax_bar, bar_title in [(ax_true, "True Groups"), (ax_sel, "Selected Groups")]:
        ax_bar.set_facecolor("#f7f7f7")
        ax_bar.set_yticks([0], [bar_title])
        ax_bar.tick_params(axis="y", length=0, labelsize=10)
        ax_bar.set_xticks(range(K), [labels[i] for i in leaf_order], fontsize=10)
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#666666")
            spine.set_linewidth(0.8)

    ax_true.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_sel.set_xlabel("Treatment (leaf order)", fontsize=12)

    ax_tree.text(
        0.99,
        0.02,
        "Top: selected-cycle path summary\nBottom bars: true vs final selected grouping",
        transform=ax_tree.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return save_path

# =========================================================
# dendrogram
# =========================================================

def plot_scaf_dendrogram(
    lambda_values,
    all_Z_list,
    round_decimals: int = 2,
    title: str = "",
    save_path: str = "dendrogram.png",
    snap_eps: float | None = None,
    labels=None,
    xlabel: str = "Treatment Index",
    picked_lambda_index: int | None = None,
):
    """
    Build a path-based dendrogram from the sequence of Z matrices.

    Important:
    - y-axis is lambda / fusion height
    - selected lambda is shown as a horizontal dashed line
    """
    if not all_Z_list:
        return None

    L = len(all_Z_list)
    K = all_Z_list[0].shape[0]

    all_clusterings = []
    for Z in all_Z_list:
        if Z.shape[0] != K:
            raise ValueError("Inconsistent K across all_Z_list.")
        if snap_eps is None:
            cid = _cluster_ids_by_round(Z, round_decimals)
        else:
            cid = _cluster_ids_by_eps(Z, float(snap_eps))
        all_clusterings.append(cid)

    linkage_list = []
    cluster_members = {i: frozenset([i]) for i in range(K)}
    cluster_heights = {i: 0.0 for i in range(K)}
    next_id = K
    merges_found = False

    for t in range(1, L):
        lam = float(lambda_values[t])
        groups = all_clusterings[t]

        merger_map = {}
        active = list(cluster_members.keys())

        for cid in active:
            item = next(iter(cluster_members[cid]))
            new_gid = int(groups[item])
            merger_map.setdefault(new_gid, set()).add(cid)

        for _, to_merge in merger_map.items():
            if len(to_merge) <= 1:
                continue
            merges_found = True

            lst = sorted(list(to_merge), key=lambda x: cluster_heights[x])

            while len(lst) > 1:
                a = lst.pop(0)
                b = lst.pop(0)

                new_id = next_id
                new_h = lam
                new_mem = cluster_members[a].union(cluster_members[b])

                linkage_list.append([a, b, new_h, len(new_mem)])

                del cluster_members[a]
                del cluster_members[b]
                cluster_members[new_id] = new_mem
                cluster_heights[new_id] = new_h
                next_id += 1

                lst.append(new_id)
                lst.sort(key=lambda x: cluster_heights[x])

    if (not linkage_list) or (not merges_found):
        plt.figure(figsize=(8, 4))
        plt.title(title if title else "Dendrogram")
        plt.text(
            0.5, 0.5,
            "No merges found along lambda path",
            ha="center", va="center", fontsize=12
        )
        plt.xlabel(xlabel)
        plt.ylabel(r"$\lambda$ (fusion strength)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        return save_path

    # complete to full tree if needed
    if len(linkage_list) < K - 1:
        last_height = float(max(link[2] for link in linkage_list)) if linkage_list else 0.0
        eps = max(1e-8, 1e-6 * max(1.0, last_height))

        remaining = sorted(cluster_members.keys(), key=lambda x: cluster_heights[x])
        while len(remaining) > 1:
            a = remaining.pop(0)
            b = remaining.pop(0)

            new_id = next_id
            new_h = max(cluster_heights[a], cluster_heights[b], last_height) + eps
            new_mem = cluster_members[a].union(cluster_members[b])

            linkage_list.append([a, b, new_h, len(new_mem)])

            del cluster_members[a]
            del cluster_members[b]
            cluster_members[new_id] = new_mem
            cluster_heights[new_id] = new_h
            next_id += 1

            remaining.append(new_id)
            remaining.sort(key=lambda x: cluster_heights[x])

    Z_link = np.array(linkage_list, dtype=float)

    if Z_link.shape[0] != K - 1:
        raise ValueError(
            f"Invalid linkage matrix: got {Z_link.shape[0]} merges, expected {K-1}."
        )

    if labels is None:
        labels = [str(i) for i in range(K)]
    else:
        if len(labels) != K:
            raise ValueError("labels length must equal K.")

    print(f"[Dendrogram] K={K}, len(labels)={len(labels)}, merges={len(linkage_list)}")

    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambda_values):
        picked_lambda = float(lambda_values[picked_lambda_index])
        title = (
            f"{title}\n"
            f"(selected lambda idx={picked_lambda_index}, lambda={picked_lambda:.4g})"
        )

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(r"$\lambda$ (fusion strength)")

    sch.dendrogram(Z_link, labels=labels, leaf_rotation=0.0, leaf_font_size=10)

    # horizontal dashed line for selected lambda
    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambda_values):
        picked_lambda = float(lambda_values[picked_lambda_index])
        plt.axhline(
            y=picked_lambda,
            linestyle="--",
            linewidth=1.5,
            label=f"Selected λ={picked_lambda:.4g}"
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path
