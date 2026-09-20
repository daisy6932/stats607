import torch


def _normalize(pen: torch.Tensor, C: int, d: int, normalize: bool) -> torch.Tensor:
    if normalize and C > 1 and d > 0:
        pen = pen / (C * (C - 1) / 2.0) / d
    return pen


def _pairwise_diffs(beta: torch.Tensor) -> torch.Tensor:
    """
    Return all pairwise differences beta[q]-beta[k] for q<k as a tensor of shape (P, d),
    where P = C*(C-1)/2.
    """
    C, d = beta.shape
    idx_q, idx_k = torch.triu_indices(C, C, offset=1, device=beta.device)
    diffs = beta[idx_q] - beta[idx_k]   # (P, d)
    return diffs


def fusion_penalty_l1(beta: torch.Tensor, lam: float, normalize: bool = False) -> torch.Tensor:
    C, d = beta.shape
    diffs = _pairwise_diffs(beta)
    pen = lam * torch.abs(diffs).sum()
    return _normalize(pen, C, d, normalize)


def fusion_penalty(
    beta: torch.Tensor,
    lam: float,
    normalize: bool = False
) -> torch.Tensor:
    """Coordinatewise pairwise L1 fusion penalty used by NSCF."""
    return fusion_penalty_l1(beta, lam, normalize)
