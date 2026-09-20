import torch

from its_nn_value_first_grouped.data import (
    _baseline_function,
    _parse_float_list,
    _parse_group_prototypes_spec,
    compute_true_mu_and_opt_actions,
    generate_synthetic_data,
)


def generate_synthetic_data_with_truth(
    n_samples,
    n_features,
    n_treatments,
    groups,
    seed,
    scenario="nonlinear_easy",
    sigma=0.5,
    treat_random=True,
    baseline_scale=1.0,
    effect_scale=1.0,
    near_fused_eps=0.0,
    near_fused_seed=None,
    group_effect_style="auto",
    baseline_style="auto",
    treatment_prob_style="auto",
    group_prototypes=None,
    treatment_probs=None,
    ensure_all_treatments=False,
):
    if isinstance(group_prototypes, str):
        group_prototypes = _parse_group_prototypes_spec(group_prototypes)
    if isinstance(treatment_probs, str):
        treatment_probs = _parse_float_list(treatment_probs)

    X, A, y, gamma_true = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_treatments=n_treatments,
        groups=groups,
        seed=seed,
        scenario=scenario,
        sigma=sigma,
        treat_random=treat_random,
        baseline_scale=baseline_scale,
        effect_scale=effect_scale,
        near_fused_eps=near_fused_eps,
        near_fused_seed=near_fused_seed,
        group_effect_style=group_effect_style,
        baseline_style=baseline_style,
        treatment_prob_style=treatment_prob_style,
        group_prototypes=group_prototypes,
        treatment_probs=treatment_probs,
        ensure_all_treatments=ensure_all_treatments,
    )

    mu_true, a_opt = compute_true_mu_and_opt_actions(
        X=X,
        gamma_true=gamma_true,
        scenario=scenario,
        baseline_scale=baseline_scale,
        baseline_style=baseline_style,
        group_effect_style=group_effect_style,
    )
    m_true = _baseline_function(
        X,
        scenario=scenario,
        baseline_scale=baseline_scale,
        baseline_style=baseline_style,
    )
    return {
        "X": X.float(),
        "A": A.long(),
        "y": y.float(),
        "gamma_true": gamma_true.float(),
        "mu_true": mu_true.float(),
        "a_opt": a_opt.long(),
        "m_true": m_true.float(),
    }
