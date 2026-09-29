import math
from typing import List, Optional, Sequence, Tuple

import torch


GROUPED_SCENARIOS = {
    "grouped_linear",
    "grouped_nonlinear_easy",
    "grouped_nonlinear_hard",
}


def _phi_linear(X: torch.Tensor) -> torch.Tensor:
    return X


def _phi_reference_linear(X: torch.Tensor) -> torch.Tensor:
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    ones = torch.ones_like(x1)
    return torch.cat([ones, x1, x2], dim=1)


def _phi_nonlinear_basic(X: torch.Tensor) -> torch.Tensor:
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x3 = X[:, 2:3]
    feats = [
        x1,
        x2,
        x3,
        x1**2,
        x2**2,
        torch.sin(math.pi * x1),
        x1 * x2,
    ]
    return torch.cat(feats, dim=1)


def _phi_nonlinear_richer(X: torch.Tensor) -> torch.Tensor:
    n, p = X.shape
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x3 = X[:, 2:3]
    x4 = X[:, 3:4] if p >= 4 else X[:, 0:1]
    x5 = X[:, 4:5] if p >= 5 else X[:, 1:2]
    feats = [
        x1,
        x2,
        x3,
        x1**2,
        x2**2,
        x3**2,
        torch.sin(math.pi * x1),
        torch.cos(math.pi * x2),
        x1 * x2,
        x2 * x3,
        x4 * x5,
    ]
    return torch.cat(feats, dim=1)


def _scenario_base_name(scenario: str) -> str:
    if scenario == "grouped_linear":
        return "linear"
    if scenario in {"reference_nonlinear_10tr2g", "reference_nonlinear_strong_10tr2g"}:
        return "nonlinear_easy"
    if scenario == "grouped_nonlinear_easy":
        return "nonlinear_easy"
    if scenario == "grouped_nonlinear_hard":
        return "nonlinear_hard"
    return scenario


def _select_phi(
    X: torch.Tensor,
    scenario: str,
    baseline_style: str = "auto",
    group_effect_style: str = "auto",
) -> torch.Tensor:
    base = _scenario_base_name(scenario)
    if scenario == "reference_linear_10tr2g":
        return _phi_reference_linear(X[:, : min(3, X.shape[1])])
    if scenario == "grouped_linear":
        return _phi_reference_linear(X[:, : min(3, X.shape[1])])
    if base in ["linear_easy", "linear_baseline"]:
        return _phi_linear(X[:, : min(3, X.shape[1])])
    if base in ["nonlinear_easy", "nonlinear_moderate", "observational"]:
        return _phi_nonlinear_basic(X)
    if base in ["nonlinear_hard", "misspecified"]:
        return _phi_nonlinear_richer(X)
    raise ValueError(f"Unknown scenario={scenario}")


def _baseline_function(
    X: torch.Tensor,
    scenario: str,
    baseline_scale: float,
    baseline_style: str = "auto",
) -> torch.Tensor:
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    base = _scenario_base_name(scenario)

    if baseline_style == "nonlinear":
        m = 0.6 * torch.cos(math.pi * x1) + 0.4 * (x2**2) - 0.3 * x3
        return baseline_scale * m

    use_reference = baseline_style == "reference" or scenario in {
        "reference_linear_10tr2g",
        "reference_nonlinear_10tr2g",
        "reference_nonlinear_strong_10tr2g",
    }
    if use_reference:
        return 1.0 + 2.0 * x1 + x2 + 0.5 * x3

    if base in ["linear_easy", "linear_baseline", "linear"]:
        m = 0.8 * x1 - 0.5 * x2 + 0.3 * x3
    elif base in ["nonlinear_easy", "nonlinear_moderate", "observational", "misspecified"]:
        m = 0.6 * torch.cos(math.pi * x1) + 0.4 * (x2**2) - 0.3 * x3
    elif base == "nonlinear_hard":
        m = (
            0.7 * torch.cos(math.pi * x1)
            + 0.5 * torch.sin(math.pi * x2)
            + 0.3 * (x3**2)
            - 0.2 * x1 * x2
        )
    else:
        raise ValueError(f"Unknown scenario={scenario}")

    return baseline_scale * m


def _parse_float_list(spec: Optional[str]) -> Optional[List[float]]:
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_group_prototypes_spec(spec: Optional[str]) -> Optional[List[List[float]]]:
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None
    rows = []
    for block in spec.split("|"):
        vals = [float(x.strip()) for x in block.split(",") if x.strip()]
        if vals:
            rows.append(vals)
    return rows or None


def _reference_group_prototype_bank(num_groups: int) -> Optional[List[List[float]]]:
    if num_groups == 2:
        return [
            [-0.36, 1.8, 1.8],
            [0.36, -1.8, -1.8],
        ]
    if num_groups == 3:
        return [
            [0.5, 1.0, -1.0],
            [1.0, 0.0, 1.0],
            [0.5, -1.0, 0.0],
        ]
    if num_groups == 5:
        return [
            [0.5, 0.5, 1.5],
            [1.0, -2.0, -1.0],
            [-1.0, 1.0, -2.0],
            [-2.0, -1.0, 1.0],
            [1.5, 1.5, 0.5],
        ]
    return None


def _reference_nonlinear_10tr2g_prototype_bank() -> List[List[float]]:
    # Coordinates match _phi_nonlinear_basic:
    # x1, x2, x3, x1^2, x2^2, sin(pi*x1), x1*x2.
    return [
        [-0.36, 1.8, 1.8, 1.1, -0.9, 1.25, -1.0],
        [0.36, -1.8, -1.8, -1.1, 0.9, -1.25, 1.0],
    ]


def _reference_nonlinear_strong_10tr2g_prototype_bank() -> List[List[float]]:
    # Coordinates match _phi_nonlinear_basic:
    # x1, x2, x3, x1^2, x2^2, sin(pi*x1), x1*x2.
    # Linear terms are intentionally weak; nonlinear signatures are stronger
    # and not exact mirror images.
    return [
        [-0.10, 0.40, 0.35, 1.50, -1.40, 1.80, -1.60],
        [0.20, -0.35, -0.55, -1.20, 1.70, -1.45, 1.95],
    ]


def _generic_group_prototypes(
    num_groups: int,
    d_phi: int,
    base: str,
    effect_scale: float,
) -> List[List[float]]:
    centers = torch.linspace(-1.0, 1.0, steps=num_groups)
    protos: List[List[float]] = []
    for gi, c in enumerate(centers.tolist()):
        vec = torch.zeros(d_phi)
        alt = -1.0 if gi % 2 else 1.0
        if base == "linear":
            k = min(3, d_phi)
            vals = torch.tensor(
                [
                    0.6 * c,
                    1.4 * c + 0.35 * alt,
                    -1.2 * c + 0.25 * math.sin((gi + 1) * math.pi / max(num_groups, 2)),
                ]
            )[:k]
            vec[:k] = vals
        elif base in {"nonlinear_easy", "nonlinear_moderate"}:
            k = min(7, d_phi)
            vals = torch.tensor(
                [
                    2.0 * c,
                    1.4 * c + 0.4 * alt,
                    -1.3 * c,
                    1.2 * math.sin((gi + 1) * math.pi / max(num_groups, 2)),
                    0.8 * math.cos((gi + 1) * math.pi / max(num_groups, 2)),
                    0.9 * alt * (1.0 - abs(c) / 2.0),
                    1.1 * c * alt,
                ]
            )[:k]
            vec[:k] = vals
        elif base == "nonlinear_hard":
            k = min(11, d_phi)
            vals = torch.tensor(
                [
                    2.4 * c,
                    1.8 * c + 0.5 * alt,
                    -1.6 * c,
                    1.4 * math.sin((gi + 1) * math.pi / max(num_groups, 2)),
                    1.0 * math.cos((gi + 1) * math.pi / max(num_groups, 2)),
                    1.2 * alt * (1.0 - abs(c) / 2.0),
                    1.4 * c * alt,
                    -0.9 * alt,
                    1.1 * c + 0.25 * alt,
                    -0.8 * c + 0.15 * alt,
                    0.9 * math.sin((gi + 1) * math.pi / (num_groups + 1)),
                ]
            )[:k]
            vec[:k] = vals
        else:
            raise ValueError(f"Unknown grouped base={base}")
        protos.append((effect_scale * vec).tolist())
    return protos


def _resolve_group_prototypes(
    d_phi: int,
    groups: List[List[int]],
    scenario: str,
    effect_scale: float,
    group_effect_style: str = "auto",
    group_prototypes: Optional[Sequence[Sequence[float]]] = None,
) -> List[torch.Tensor]:
    num_groups = len(groups)
    base = _scenario_base_name(scenario)

    if scenario == "reference_linear_10tr2g":
        raw = _reference_group_prototype_bank(2)
        assert raw is not None
        out = []
        for row in raw:
            vec = torch.zeros(d_phi)
            vec[: min(len(row), d_phi)] = torch.tensor(row[: min(len(row), d_phi)])
            out.append(vec)
        return out
    if scenario == "reference_nonlinear_10tr2g":
        raw = _reference_nonlinear_10tr2g_prototype_bank()
        out = []
        for row in raw:
            vec = torch.zeros(d_phi)
            vals = [effect_scale * v for v in row]
            vec[: min(len(vals), d_phi)] = torch.tensor(vals[: min(len(vals), d_phi)])
            out.append(vec)
        return out
    if scenario == "reference_nonlinear_strong_10tr2g":
        raw = _reference_nonlinear_strong_10tr2g_prototype_bank()
        out = []
        for row in raw:
            vec = torch.zeros(d_phi)
            vals = [effect_scale * v for v in row]
            vec[: min(len(vals), d_phi)] = torch.tensor(vals[: min(len(vals), d_phi)])
            out.append(vec)
        return out

    if group_prototypes is not None:
        if len(group_prototypes) != num_groups:
            raise ValueError("group_prototypes must have one row per true group")
        out = []
        for row in group_prototypes:
            vec = torch.zeros(d_phi)
            vals = list(row)
            vec[: min(len(vals), d_phi)] = torch.tensor(vals[: min(len(vals), d_phi)])
            out.append(vec)
        return out

    if group_effect_style == "reference":
        raw = _reference_group_prototype_bank(num_groups)
        if raw is None:
            raise ValueError(
                "reference group_effect_style currently supports only 2, 3, or 5 true groups"
            )
        out = []
        for row in raw:
            vec = torch.zeros(d_phi)
            vec[: min(len(row), d_phi)] = effect_scale * torch.tensor(row[: min(len(row), d_phi)])
            out.append(vec)
        return out

    if base in ["linear_easy", "linear_baseline", "linear"]:
        raw = _generic_group_prototypes(num_groups, d_phi, "linear", effect_scale)
    elif base in ["nonlinear_easy", "nonlinear_moderate", "observational", "misspecified"]:
        raw = _generic_group_prototypes(num_groups, d_phi, "nonlinear_moderate", effect_scale)
    elif base == "nonlinear_hard":
        raw = _generic_group_prototypes(num_groups, d_phi, "nonlinear_hard", effect_scale)
    else:
        raise ValueError(f"Unknown scenario={scenario}")

    out = []
    for row in raw:
        vec = torch.zeros(d_phi)
        vals = list(row)
        vec[: min(len(vals), d_phi)] = torch.tensor(vals[: min(len(vals), d_phi)])
        out.append(vec)
    return out


def _make_true_gamma(
    n_treatments: int,
    d_phi: int,
    groups: List[List[int]],
    scenario: str,
    effect_scale: float,
    near_fused_eps: float = 0.0,
    near_fused_seed: Optional[int] = None,
    group_effect_style: str = "auto",
    group_prototypes: Optional[Sequence[Sequence[float]]] = None,
) -> torch.Tensor:
    gamma_true = torch.zeros(n_treatments, d_phi)
    prototypes = _resolve_group_prototypes(
        d_phi=d_phi,
        groups=groups,
        scenario=scenario,
        effect_scale=effect_scale,
        group_effect_style=group_effect_style,
        group_prototypes=group_prototypes,
    )

    perturb_generator = None
    if near_fused_eps > 0 and near_fused_seed is not None:
        perturb_generator = torch.Generator(device=gamma_true.device)
        perturb_generator.manual_seed(int(near_fused_seed))

    for gi, grp in enumerate(groups):
        proto = prototypes[gi]
        for t in grp:
            gamma_true[t] = proto.clone()
            if near_fused_eps > 0:
                perturbation = torch.randn(
                    d_phi,
                    generator=perturb_generator,
                    device=gamma_true.device,
                    dtype=gamma_true.dtype,
                )
                gamma_true[t] += near_fused_eps * perturbation
    return gamma_true


def _default_treatment_probs(n_treatments: int, style: str) -> List[float]:
    if style == "uniform":
        return [1.0 / n_treatments] * n_treatments
    if style == "reference_unbalanced":
        if n_treatments == 10:
            return [0.2, 0.15, 0.05, 0.05, 0.05, 0.2, 0.1, 0.1, 0.05, 0.05]
        if n_treatments == 15:
            return (
                [0.05, 0.05, 0.05, 1.0 / 6.0 - 0.15 / 2.0, 1.0 / 6.0 - 0.15 / 2.0]
                + [0.05, 0.05, 0.05, 1.0 / 6.0 - 0.15 / 2.0, 1.0 / 6.0 - 0.15 / 2.0]
                + [0.05, 0.05, 1.0 / 9.0 - 0.1 / 3.0, 1.0 / 9.0 - 0.1 / 3.0, 1.0 / 9.0 - 0.1 / 3.0]
            )
        if n_treatments == 20:
            return [0.2, 0.15, 0.05, 0.05, 0.05] + [0.1] * 15
        raise ValueError("reference_unbalanced currently supports n_treatments in {10, 15, 20}")
    raise ValueError(f"Unknown treatment_prob_style={style}")


def _sample_treatment(
    X: torch.Tensor,
    n_treatments: int,
    treat_random: bool,
    seed: int,
    scenario: Optional[str] = None,
    treatment_prob_style: str = "auto",
    treatment_probs: Optional[Sequence[float]] = None,
    group_effect_style: str = "auto",
) -> torch.Tensor:
    torch.manual_seed(seed + 12345)
    n, p = X.shape

    if treat_random:
        probs = None
        if treatment_probs is not None:
            probs = torch.tensor(list(treatment_probs), dtype=X.dtype, device=X.device)
        else:
            style = treatment_prob_style
            if style == "auto":
                if scenario in {
                    "reference_linear_10tr2g",
                    "reference_nonlinear_10tr2g",
                    "reference_nonlinear_strong_10tr2g",
                }:
                    style = "reference_unbalanced"
                elif scenario in GROUPED_SCENARIOS and group_effect_style == "reference":
                    style = "reference_unbalanced"
                else:
                    style = "uniform"
            probs = torch.tensor(_default_treatment_probs(n_treatments, style), dtype=X.dtype, device=X.device)
        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=n, replacement=True)

    W = torch.randn(n_treatments, p, device=X.device) * 0.35
    logits = X @ W.T
    return torch.distributions.Categorical(logits=logits).sample()


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_treatments: int,
    groups: List[List[int]],
    seed: int,
    scenario: str = "nonlinear_moderate",
    sigma: float = 0.5,
    treat_random: bool = True,
    baseline_scale: float = 1.0,
    effect_scale: float = 1.0,
    near_fused_eps: float = 0.0,
    near_fused_seed: Optional[int] = None,
    group_effect_style: str = "auto",
    baseline_style: str = "auto",
    treatment_prob_style: str = "auto",
    group_prototypes: Optional[Sequence[Sequence[float]]] = None,
    treatment_probs: Optional[Sequence[float]] = None,
    ensure_all_treatments: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if n_treatments != sum(len(g) for g in groups):
        raise ValueError("n_treatments must equal total size of groups")
    if isinstance(group_prototypes, str):
        group_prototypes = _parse_group_prototypes_spec(group_prototypes)
    if isinstance(treatment_probs, str):
        treatment_probs = _parse_float_list(treatment_probs)

    cpu = torch.device("cpu")
    torch.manual_seed(seed)
    X = -1.0 + 2.0 * torch.rand(n_samples, n_features, device=cpu)

    Phi = _select_phi(X, scenario, baseline_style=baseline_style, group_effect_style=group_effect_style)
    d_phi = Phi.shape[1]

    if _scenario_base_name(scenario) == "observational":
        treat_random = False

    A = _sample_treatment(
        X,
        n_treatments=n_treatments,
        treat_random=treat_random,
        seed=seed,
        scenario=scenario,
        treatment_prob_style=treatment_prob_style,
        treatment_probs=treatment_probs,
        group_effect_style=group_effect_style,
    )
    if ensure_all_treatments and treat_random:
        missing = [a for a in range(n_treatments) if not torch.any(A == a)]
        if missing:
            torch.manual_seed(seed + 54321)
            replace_idx = torch.randperm(n_samples, device=cpu)[: len(missing)]
            A[replace_idx] = torch.tensor(missing, dtype=A.dtype, device=cpu)

    gamma_true = _make_true_gamma(
        n_treatments=n_treatments,
        d_phi=d_phi,
        groups=groups,
        scenario=scenario,
        effect_scale=effect_scale,
        near_fused_eps=near_fused_eps,
        near_fused_seed=near_fused_seed,
        group_effect_style=group_effect_style,
        group_prototypes=group_prototypes,
    ).to(cpu)

    m = _baseline_function(X, scenario=scenario, baseline_scale=baseline_scale, baseline_style=baseline_style)
    gamma_A = gamma_true[A]
    tau = torch.sum(Phi * gamma_A, dim=1)
    eps = sigma * torch.randn(n_samples, device=cpu)
    y = m + tau + eps
    return X, A.long(), y.float(), gamma_true


def compute_true_mu_and_opt_actions(
    X: torch.Tensor,
    gamma_true: torch.Tensor,
    scenario: str,
    baseline_scale: float,
    baseline_style: str = "auto",
    group_effect_style: str = "auto",
):
    Phi = _select_phi(X, scenario, baseline_style=baseline_style, group_effect_style=group_effect_style)
    m = _baseline_function(X, scenario=scenario, baseline_scale=baseline_scale, baseline_style=baseline_style)
    mu_true = m.unsqueeze(1) + Phi @ gamma_true.T
    a_opt = torch.argmax(mu_true, dim=1)
    return mu_true.float(), a_opt.long()
