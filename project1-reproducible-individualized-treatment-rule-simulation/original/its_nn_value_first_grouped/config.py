import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # =========================================================
    # mode / output
    # =========================================================
    # Retain mode for compatibility with existing references to args.mode.
    parser.add_argument("--mode", type=str, default="alt", choices=["alt"])
    parser.add_argument("--out_base", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="exp1")
    parser.add_argument("--save_all_cycle_artifacts", action="store_true")
    parser.add_argument("--save_path_summaries", action="store_true")
    parser.add_argument(
        "--save_admm_trace",
        action="store_true",
        help="Save per-inner-epoch ADMM residual and objective diagnostics.",
    )

    # =========================================================
    # data
    # =========================================================
    parser.add_argument("--n_train", type=int, default=600)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=10000)

    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=10)

    parser.add_argument("--seed_train", type=int, default=0)
    parser.add_argument("--seed_val", type=int, default=1)
    parser.add_argument("--seed_test", type=int, default=42)

    parser.add_argument(
        "--scenario",
        type=str,
        default="nonlinear_moderate",
        choices=[
            "reference_linear_10tr2g",
            "reference_nonlinear_10tr2g",
            "reference_nonlinear_strong_10tr2g",
            "grouped_linear",
            "grouped_nonlinear_easy",
            "grouped_nonlinear_hard",
            "linear_easy",
            "linear_baseline",
            "nonlinear_easy",
            "nonlinear_moderate",
            "nonlinear_hard",
            "observational",
            "misspecified",
        ],
    )
    parser.add_argument("--y_noise", type=float, default=0.5)
    parser.add_argument("--baseline_scale", type=float, default=0.3)
    parser.add_argument("--effect_scale", type=float, default=5.0)
    parser.add_argument("--near_fused_eps", type=float, default=0.0)
    parser.add_argument(
        "--near_fused_seed",
        type=int,
        default=314159,
        help="Base seed for treatment-specific within-group perturbations; held fixed across train/validation/test splits.",
    )
    parser.add_argument(
        "--group_effect_style",
        type=str,
        default="auto",
        choices=["auto", "reference"],
        help="How true group-specific treatment effects are parameterized.",
    )
    parser.add_argument(
        "--baseline_style",
        type=str,
        default="auto",
        choices=["auto", "reference", "nonlinear"],
        help="How the baseline mean function m(x) is generated.",
    )
    parser.add_argument(
        "--treatment_prob_style",
        type=str,
        default="auto",
        choices=["auto", "uniform", "reference_unbalanced"],
        help="How randomized treatment assignment probabilities are generated.",
    )
    parser.add_argument(
        "--group_prototypes",
        type=str,
        default="",
        help="Optional true group coefficient prototypes, formatted as row1|row2|..., each row comma-separated.",
    )
    parser.add_argument(
        "--treatment_probs",
        type=str,
        default="",
        help="Optional randomized treatment probabilities, comma-separated and normalized internally.",
    )
    parser.add_argument(
        "--ensure_all_treatments",
        action="store_true",
        help="Force each treatment arm to appear at least once in generated randomized data.",
    )

    # =========================================================
    # model
    # =========================================================
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_baseline", action="store_true", default=True)

    # =========================================================
    # supervised pretrain
    # =========================================================
    parser.add_argument("--train_epochs", type=int, default=300)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--train_wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--train_improve_tol", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--scale_loss_by_y_std", action="store_true", default=True)
    parser.add_argument("--min_target_scale", type=float, default=1.0)

    # =========================================================
    # alternating
    # =========================================================
    parser.add_argument("--alt_cycles", type=int, default=3)

    # Used for the joint refit within each cycle.
    parser.add_argument("--alt_refit_epochs", type=int, default=30)
    parser.add_argument("--alt_refit_lr", type=float, default=5e-4)
    parser.add_argument("--alt_refit_wd", type=float, default=1e-4)
    parser.add_argument("--mu_refit", type=float, default=1e-4)

    parser.add_argument("--reset_U_each_cycle", action="store_true")
    parser.add_argument("--warm_start_from_Zstar", action="store_true")
    parser.add_argument("--disable_cycle_early_stop", action="store_true")

    # =========================================================
    # lambda path
    # =========================================================
    parser.add_argument("--num_lambda", type=int, default=30)
    parser.add_argument("--max_lambda", type=float, default=7.0)

    # Used primarily for cycle-stability and cluster visualizations.
    parser.add_argument("--round_decimals", type=int, default=2)

    # =========================================================
    # ADMM
    # =========================================================
    parser.add_argument("--admm_epochs", type=int, default=60)
    parser.add_argument("--theta_steps", type=int, default=30)
    parser.add_argument("--rho", type=float, default=8.0)
    parser.add_argument("--lr_gamma", type=float, default=1e-3)

    # =========================================================
    # Z solver
    # =========================================================
    parser.add_argument("--z_steps", type=int, default=150)
    parser.add_argument("--z_lr", type=float, default=3e-3)
    parser.add_argument("--z_early_tol", type=float, default=1e-4)
    parser.add_argument("--z_early_patience", type=int, default=10)
    parser.add_argument("--snap_eps", type=float, default=1e-3)
    parser.add_argument("--normalize_fusion", action="store_true")

    # =========================================================
    # fusion penalty used in the paper
    # =========================================================
    parser.add_argument("--penalty", type=str, default="l1", choices=["l1"])

    # =========================================================
    # loss
    # =========================================================
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)

    # =========================================================
    # reproducibility
    # =========================================================
    parser.add_argument("--global_seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")

    # =========================================================
    # cycle-level model selection
    # =========================================================
    parser.add_argument(
        "--cycle_metric",
        type=str,
        default="val_plus_stability",
        choices=["val", "val_plus_stability"],
    )
    parser.add_argument("--cycle_improve_tol", type=float, default=1e-3)
    parser.add_argument("--cycle_patience", type=int, default=2)

    parser.add_argument("--stability_metric", type=str, default="fro", choices=["fro", "labels"])
    parser.add_argument("--stability_tol", type=float, default=1e-3)
    parser.add_argument("--stability_weight", type=float, default=0.1)

    parser.add_argument("--save_cycle_summary", action="store_true")

    # =========================================================
    # refit-based lambda selection
    # =========================================================
    parser.add_argument(
        "--refit_lambda_rule",
        type=str,
        default="improve_eps_simplest",
        choices=[
            "min",
            "eps_simplest",
            "score",
            "improve_eps_simplest",
        ],
    )
    parser.add_argument("--refit_eps", type=float, default=0.05)
    parser.add_argument("--refit_group_penalty", type=float, default=0.0)
    parser.add_argument("--refit_improve_tol", type=float, default=0.0)
    parser.add_argument("--structure_improve_tol", type=float, default=0.0)
    # Legacy compatibility only; no longer used in no-polish selection logic.
    parser.add_argument("--preferred_n_groups", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--structure_mse_rel_tol", type=float, default=0.05)
    parser.add_argument("--structure_mse_abs_tol", type=float, default=0.0)

    parser.add_argument("--refit_partition_tol", type=float, default=1e-6)
    parser.add_argument("--refit_round_decimals", type=int, default=6)
    parser.add_argument("--dedup_partitions", action="store_true", default=True)

    # Used for grouped refitting during lambda evaluation.
    parser.add_argument("--refit_select_epochs", type=int, default=80)
    parser.add_argument("--refit_select_lr", type=float, default=1e-3)
    parser.add_argument("--refit_select_wd", type=float, default=1e-4)
    parser.add_argument("--two_group_polish_phase1_epochs", type=int, default=80)
    parser.add_argument("--two_group_polish_phase2_epochs", type=int, default=120)
    parser.add_argument("--two_group_polish_phase1_lr", type=float, default=5e-4)
    parser.add_argument("--two_group_polish_phase2_lr", type=float, default=2.5e-4)

    return parser
