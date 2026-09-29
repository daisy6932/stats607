import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch

from its_nn_value_first_grouped.config import add_args as add_clean4_args
from its_nn_value_first_grouped.model import Net
from its_nn_value_first_grouped.plotting import (
    plot_refit_lambda_selection_curve,
    plot_selected_cycle_path_with_group_bars,
    plot_solution_path_dendrogram,
    save_partition_heatmap_from_group_ids,
    save_selected_partition_dendrogram_from_group_ids,
)
from its_nn_value_first_grouped.refit_selection import eval_grouped_model_on_original_actions
from its_nn_value_first_grouped.run_scaf_server import build_lambda_path, set_global_seed
from its_nn_value_first_grouped.solver_alt import run_alternating
from its_nn_value_first_grouped.trainer import eval_mse


def _group_ids_to_groups(group_ids):
    group_ids = np.asarray(group_ids).astype(int)
    groups = []
    for gid in np.unique(group_ids):
        groups.append(np.where(group_ids == gid)[0].astype(int).tolist())
    return groups


def _pairwise_grouping_accuracy(pred_group_ids, true_group_ids):
    pred_group_ids = np.asarray(pred_group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)
    k = len(pred_group_ids)
    agree = 0
    total = 0
    for i in range(k):
        for j in range(i + 1, k):
            if (pred_group_ids[i] == pred_group_ids[j]) == (true_group_ids[i] == true_group_ids[j]):
                agree += 1
            total += 1
    return float(agree / max(total, 1))


def _exact_group_recovery(pred_group_ids, true_group_ids):
    pred_group_ids = np.asarray(pred_group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)
    k = len(pred_group_ids)
    for i in range(k):
        for j in range(i + 1, k):
            if (pred_group_ids[i] == pred_group_ids[j]) != (true_group_ids[i] == true_group_ids[j]):
                return False
    return True


def _comb2(n):
    return 0.5 * int(n) * (int(n) - 1)


def _ari(pred_group_ids, true_group_ids):
    pred_group_ids = np.asarray(pred_group_ids).astype(int)
    true_group_ids = np.asarray(true_group_ids).astype(int)
    pred_labels, pred_inverse = np.unique(pred_group_ids, return_inverse=True)
    true_labels, true_inverse = np.unique(true_group_ids, return_inverse=True)
    contingency = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    for i in range(len(pred_group_ids)):
        contingency[pred_inverse[i], true_inverse[i]] += 1
    sum_cells = float(sum(_comb2(v) for v in contingency.ravel()))
    sum_rows = float(sum(_comb2(v) for v in contingency.sum(axis=1)))
    sum_cols = float(sum(_comb2(v) for v in contingency.sum(axis=0)))
    total_pairs = float(_comb2(len(pred_group_ids)))
    if total_pairs <= 0:
        return 1.0
    expected = (sum_rows * sum_cols) / total_pairs
    max_index = 0.5 * (sum_rows + sum_cols)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0
    return float((sum_cells - expected) / denom)


def _grouping_metrics(group_ids, true_group_ids):
    return {
        "exact_recovery": bool(_exact_group_recovery(group_ids, true_group_ids)),
        "pairwise_accuracy": float(_pairwise_grouping_accuracy(group_ids, true_group_ids)),
        "ari": float(_ari(group_ids, true_group_ids)),
        "selected_groups": _group_ids_to_groups(group_ids),
    }


def _make_clean4_args(base_args):
    parser = argparse.ArgumentParser()
    add_clean4_args(parser)
    args = parser.parse_args([])
    fields = [
        "n_features", "n_actions", "n_hidden_layers", "hidden1", "hidden2", "dropout",
        "use_baseline", "train_epochs", "train_lr", "train_wd", "patience", "eval_every",
        "train_improve_tol", "grad_clip", "scale_loss_by_y_std", "min_target_scale",
        "alt_cycles", "num_lambda", "max_lambda", "round_decimals", "admm_epochs",
        "theta_steps", "rho", "lr_gamma", "z_steps", "z_lr", "z_early_tol",
        "z_early_patience", "snap_eps", "normalize_fusion", "penalty",
        "loss_type", "huber_delta", "global_seed", "deterministic",
        "save_admm_trace",
        "cycle_metric", "cycle_improve_tol", "cycle_patience", "stability_metric",
        "stability_tol", "stability_weight", "refit_lambda_rule", "refit_eps",
        "refit_group_penalty", "refit_improve_tol", "structure_improve_tol",
        "structure_mse_rel_tol", "structure_mse_abs_tol",
        "refit_partition_tol", "refit_round_decimals", "dedup_partitions",
        "refit_select_epochs", "refit_select_lr", "refit_select_wd",
        "two_group_polish_phase1_epochs", "two_group_polish_phase2_epochs",
        "two_group_polish_phase1_lr", "two_group_polish_phase2_lr",
    ]
    for key in fields:
        if hasattr(base_args, key):
            setattr(args, key, getattr(base_args, key))
    return args


class ValueFirstPolicy:
    def __init__(self, args, true_group_ids, outdir=None):
        self.args = _make_clean4_args(args)
        self.true_group_ids = np.asarray(true_group_ids).astype(int)
        self.outdir = Path(outdir) if outdir is not None else None
        self.final_model_ = None
        self.final_group_ids_ = None
        self.final_source_ = None
        self.diagnostics_ = {}

    def _candidate_key(self, cand):
        return (
            float(cand["val_mse"]),
            int(cand["n_groups"]),
            float(cand["test_mse"]),
            0 if str(cand["source"]) == "structure_best" else 1,
        )

    def fit(self, X, A, Y, X_val, A_val, Y_val):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_global_seed(int(getattr(self.args, "global_seed", 123)), deterministic=bool(getattr(self.args, "deterministic", False)))
        model = Net(
            n_features=self.args.n_features,
            n_treatments=self.args.n_actions,
            hidden1=self.args.hidden1,
            hidden2=self.args.hidden2,
            dropout=self.args.dropout,
            use_baseline=self.args.use_baseline,
            n_hidden_layers=self.args.n_hidden_layers,
        ).to(device)

        X_train_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        A_train_t = torch.tensor(np.asarray(A, dtype=np.int64), device=device)
        Y_train_t = torch.tensor(np.asarray(Y, dtype=np.float32), device=device)
        X_val_t = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
        A_val_t = torch.tensor(np.asarray(A_val, dtype=np.int64), device=device)
        Y_val_t = torch.tensor(np.asarray(Y_val, dtype=np.float32), device=device)

        lambda_values = build_lambda_path(self.args.num_lambda, self.args.max_lambda)
        if self.outdir is not None:
            self.outdir.mkdir(parents=True, exist_ok=True)

        alt_out = run_alternating(
            model=model,
            X_train=X_train_t, A_train=A_train_t, y_train=Y_train_t,
            X_val=X_val_t, A_val=A_val_t, y_val=Y_val_t,
            X_test=X_val_t, A_test=A_val_t, y_test=Y_val_t,
            true_group_ids=self.true_group_ids,
            lambda_values=lambda_values,
            args=self.args,
            log_fn=(lambda *args, **kwargs: None),
            outdir=str(self.outdir) if self.outdir is not None else None,
        )

        candidates = []
        prediction_group_ids = np.asarray(alt_out["best_group_ids"]).astype(int)
        pred_test_proxy = float(eval_grouped_model_on_original_actions(
            alt_out["best_model"], eval_mse, X_val_t, A_val_t, Y_val_t, prediction_group_ids
        ))
        candidates.append({
            "source": "prediction_best",
            "model": alt_out["best_model"],
            "group_ids": prediction_group_ids,
            "val_mse": float(alt_out["best_val_mse"]),
            "test_mse": pred_test_proxy,
            "n_groups": int(alt_out["best_picked_n_groups"]),
        })

        structure_best_rec = alt_out.get("structure_best_cycle_record")
        if structure_best_rec is not None:
            structure_group_ids = np.asarray(structure_best_rec["picked_group_ids"]).astype(int)
            structure_test_proxy = float(eval_grouped_model_on_original_actions(
                structure_best_rec["grouped_model"], eval_mse, X_val_t, A_val_t, Y_val_t, structure_group_ids
            ))
            candidates.append({
                "source": "structure_best",
                "model": structure_best_rec["grouped_model"],
                "group_ids": structure_group_ids,
                "val_mse": float(structure_best_rec["cycle_val_mse"]),
                "test_mse": structure_test_proxy,
                "n_groups": int(structure_best_rec["picked_n_groups"]),
            })

        best = min(candidates, key=self._candidate_key)
        self.final_model_ = best["model"]
        self.final_group_ids_ = np.asarray(best["group_ids"]).astype(int)
        self.final_source_ = best["source"]
        metrics = _grouping_metrics(self.final_group_ids_, self.true_group_ids)
        plot_paths = self._save_final_plots(
            lambda_values=lambda_values,
            alt_out=alt_out,
            final_source=self.final_source_,
            final_group_ids=self.final_group_ids_,
        )
        self.diagnostics_ = {
            "value_first_final_source": self.final_source_,
            "value_first_final_val_mse": float(best["val_mse"]),
            "value_first_final_test_proxy_mse": float(best["test_mse"]),
            "value_first_final_n_groups": int(best["n_groups"]),
            "value_first_exact_recovery": metrics["exact_recovery"],
            "value_first_pairwise_accuracy": metrics["pairwise_accuracy"],
            "value_first_ari": metrics["ari"],
            "value_first_final_partition_plot": plot_paths.get("selected_partition_plot"),
            "value_first_final_dendrogram_plot": plot_paths.get("selected_refit_dendrogram_plot"),
            "value_first_final_path_plot": plot_paths.get("path_with_groups_plot"),
        }
        return self

    def _save_final_plots(self, *, lambda_values, alt_out, final_source, final_group_ids):
        if self.outdir is None or final_group_ids is None:
            return {}

        self.outdir.mkdir(parents=True, exist_ok=True)
        final_group_ids = np.asarray(final_group_ids).astype(int)
        best_cycle = int(alt_out.get("best_cycle_id", 1))
        best_lambda_idx = alt_out.get("best_picked_lambda_index", None)
        best_lambda_val = alt_out.get("best_picked_lambda_value", None)
        best_n_groups = int(np.unique(final_group_ids).size)
        cycles = alt_out.get("cycles", [])
        ref_cycle = best_cycle
        ref_cycle_pack = cycles[best_cycle - 1] if cycles and best_cycle >= 1 else None

        structure_best_rec = alt_out.get("structure_best_cycle_record", None)
        if str(final_source) == "structure_best" and structure_best_rec is not None and cycles:
            ref_cycle = int(structure_best_rec["cycle"])
            ref_cycle_pack = cycles[ref_cycle - 1]
            best_lambda_idx = int(structure_best_rec.get("picked_lambda_index"))
            best_lambda_val = float(structure_best_rec.get("picked_lambda_value"))

        partition_file = self.outdir / "final_recommended_partition.png"
        save_partition_heatmap_from_group_ids(
            final_group_ids,
            str(partition_file),
            title=f"Final Recommended Partition | source={final_source}, groups={best_n_groups}",
        )

        prediction_partition_file = None
        prediction_dendro_file = None
        prediction_path_file = None
        prediction_tradeoff_file = None
        prediction_path_summary_file = None
        prediction_best_group_ids = alt_out.get("best_group_ids", None)
        if prediction_best_group_ids is not None:
            prediction_best_group_ids = np.asarray(prediction_best_group_ids).astype(int)
            prediction_partition_file = self.outdir / "prediction_best_selected_partition.png"
            save_partition_heatmap_from_group_ids(
                prediction_best_group_ids,
                str(prediction_partition_file),
                title=(
                    f"Prediction-Best Selected Partition | cycle={best_cycle}, "
                    f"lambda={best_lambda_val}, groups={int(np.unique(prediction_best_group_ids).size)}"
                ),
            )
            prediction_dendro_file = self.outdir / "prediction_best_selected_refit_dendrogram.png"
            save_selected_partition_dendrogram_from_group_ids(
                prediction_best_group_ids,
                str(prediction_dendro_file),
                title=(
                    f"Prediction-Best Refit Dendrogram | cycle={best_cycle}, "
                    f"lambda={best_lambda_val}, groups={int(np.unique(prediction_best_group_ids).size)}"
                ),
            )
            best_cycle_pack = cycles[best_cycle - 1] if cycles and best_cycle >= 1 else None
            if best_cycle_pack is not None and best_cycle_pack.get("all_Z") is not None:
                prediction_tradeoff_file = self.outdir / "prediction_best_lambda_tradeoff.png"
                plot_refit_lambda_selection_curve(
                    lambda_values=lambda_values,
                    refit_results=best_cycle_pack.get("refit_results", None),
                    picked_lambda_index=best_cycle_pack.get("picked_lambda_index", None),
                    save_path=str(prediction_tradeoff_file),
                    old_metrics=best_cycle_pack.get("admm_metrics", None),
                )
                if bool(getattr(self.args, "save_path_summaries", False)):
                    prediction_path_summary_file = self.outdir / "prediction_best_solution_path_summary.png"
                    plot_solution_path_dendrogram(
                        lambda_values=lambda_values,
                        all_Z_list=best_cycle_pack["all_Z"],
                        picked_lambda_index=best_cycle_pack.get("picked_lambda_index", None),
                        save_path=str(prediction_path_summary_file),
                        round_decimals=int(getattr(self.args, "round_decimals", 6)),
                        snap_eps=float(getattr(self.args, "snap_eps", 0.0)) if getattr(self.args, "snap_eps", None) is not None else None,
                        labels=[str(i + 1) for i in range(int(self.args.n_actions))],
                        xlabel="Treatment",
                        ylabel=r"$\log(\lambda \times 1000)$",
                        title=f"Prediction-Best Cycle {best_cycle} Path Summary",
                        lambda_scale=1000.0,
                    )
                prediction_path_file = self.outdir / "prediction_best_cycle_path_with_groups.png"
                plot_selected_cycle_path_with_group_bars(
                    lambda_values=lambda_values,
                    all_Z_list=best_cycle_pack["all_Z"],
                    picked_lambda_index=(int(best_lambda_idx) if best_lambda_idx is not None else None),
                    selected_group_ids=prediction_best_group_ids,
                    true_group_ids=self.true_group_ids,
                    save_path=str(prediction_path_file),
                    round_decimals=int(getattr(self.args, "round_decimals", 6)),
                    snap_eps=float(getattr(self.args, "snap_eps", 0.0)) if getattr(self.args, "snap_eps", None) is not None else None,
                    labels=[str(i + 1) for i in range(int(self.args.n_actions))],
                    title=(
                        f"Prediction-Best | cycle={best_cycle}, "
                        f"lambda={best_lambda_val}, groups={int(np.unique(prediction_best_group_ids).size)}"
                    ),
                )

        dendro_file = self.outdir / "final_recommended_refit_dendrogram.png"
        save_selected_partition_dendrogram_from_group_ids(
            final_group_ids,
            str(dendro_file),
            title=f"Final Recommended Dendrogram | source={final_source}, groups={best_n_groups}",
        )

        path_file = None
        final_tradeoff_file = None
        final_path_summary_file = None
        if ref_cycle_pack is not None and ref_cycle_pack.get("all_Z") is not None:
            final_tradeoff_file = self.outdir / "final_recommended_lambda_tradeoff.png"
            plot_refit_lambda_selection_curve(
                lambda_values=lambda_values,
                refit_results=ref_cycle_pack.get("refit_results", None),
                picked_lambda_index=(int(best_lambda_idx) if best_lambda_idx is not None else None),
                save_path=str(final_tradeoff_file),
                old_metrics=ref_cycle_pack.get("admm_metrics", None),
            )
            if bool(getattr(self.args, "save_path_summaries", False)):
                final_path_summary_file = self.outdir / "final_recommended_solution_path_summary.png"
                plot_solution_path_dendrogram(
                    lambda_values=lambda_values,
                    all_Z_list=ref_cycle_pack["all_Z"],
                    picked_lambda_index=(int(best_lambda_idx) if best_lambda_idx is not None else None),
                    save_path=str(final_path_summary_file),
                    round_decimals=int(getattr(self.args, "round_decimals", 6)),
                    snap_eps=float(getattr(self.args, "snap_eps", 0.0)) if getattr(self.args, "snap_eps", None) is not None else None,
                    labels=[str(i + 1) for i in range(int(self.args.n_actions))],
                    xlabel="Treatment",
                    ylabel=r"$\log(\lambda \times 1000)$",
                    title=f"Final Recommended Cycle {ref_cycle} Path Summary",
                    lambda_scale=1000.0,
                )
            path_file = self.outdir / "final_recommended_path_with_groups.png"
            plot_selected_cycle_path_with_group_bars(
                lambda_values=lambda_values,
                all_Z_list=ref_cycle_pack["all_Z"],
                picked_lambda_index=(int(best_lambda_idx) if best_lambda_idx is not None else None),
                selected_group_ids=final_group_ids,
                true_group_ids=self.true_group_ids,
                save_path=str(path_file),
                round_decimals=int(getattr(self.args, "round_decimals", 6)),
                snap_eps=float(getattr(self.args, "snap_eps", 0.0)) if getattr(self.args, "snap_eps", None) is not None else None,
                labels=[str(i + 1) for i in range(int(self.args.n_actions))],
                title=(
                    f"Final Recommended | source={final_source}, "
                    f"cycle={ref_cycle}, lambda={best_lambda_val}, groups={best_n_groups}"
                ),
            )

        return {
            "prediction_best_partition_plot": (str(prediction_partition_file) if prediction_partition_file is not None else None),
            "prediction_best_selected_refit_dendrogram_plot": (str(prediction_dendro_file) if prediction_dendro_file is not None else None),
            "prediction_best_lambda_tradeoff_plot": (str(prediction_tradeoff_file) if prediction_tradeoff_file is not None else None),
            "prediction_best_solution_path_summary_plot": (str(prediction_path_summary_file) if prediction_path_summary_file is not None else None),
            "prediction_best_path_plot": (str(prediction_path_file) if prediction_path_file is not None else None),
            "selected_partition_plot": str(partition_file),
            "selected_refit_dendrogram_plot": str(dendro_file),
            "selected_lambda_tradeoff_plot": (str(final_tradeoff_file) if final_tradeoff_file is not None else None),
            "selected_solution_path_summary_plot": (str(final_path_summary_file) if final_path_summary_file is not None else None),
            "path_with_groups_plot": (str(path_file) if path_file is not None else None),
        }

    def predict_mu(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        device = next(self.final_model_.parameters()).device
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            grouped_mu = self.final_model_.predict_all_treatments(X_t).detach().cpu().numpy()
        return grouped_mu[:, self.final_group_ids_]

    def predict_treatment(self, X):
        return self.predict_mu(X).argmax(axis=1)

    def get_diagnostics(self):
        return dict(self.diagnostics_)
