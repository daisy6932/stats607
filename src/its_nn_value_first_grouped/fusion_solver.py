import torch
from .penalty import fusion_penalty


def snap_rows(Z: torch.Tensor, snap_eps: float) -> torch.Tensor:
    """
    Snap nearly identical rows in Z (Linfty distance < snap_eps) by averaging.
    Pure torch version (no numpy round-trip).
    """
    if snap_eps <= 0:
        return Z.detach()

    Z2 = Z.detach().clone()
    C = Z2.shape[0]
    used = torch.zeros(C, dtype=torch.bool, device=Z2.device)

    for i in range(C):
        if used[i]:
            continue
        diff = (Z2 - Z2[i]).abs().amax(dim=1)  # Linfty distance
        group_mask = (diff < snap_eps) & (~used)
        idx = torch.where(group_mask)[0]
        if idx.numel() > 1:
            mean_row = Z2[idx].mean(dim=0, keepdim=True)
            Z2[idx] = mean_row
        used[idx] = True

    return Z2


def solve_Z_update(
    W: torch.Tensor,
    lam: float,
    rho: float,
    z_steps: int,
    z_lr: float,
    normalize_fusion: bool,
    early_tol: float,
    early_patience: int,
    snap_eps: float
) -> torch.Tensor:
    """
    Approx solve:
      Z = argmin_Z  fusion_penalty(Z) + (rho/2)||Z-W||^2
    using finite Adam steps on Z.
    """
    Z = W.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=z_lr, amsgrad=True)

    last = None
    stall = 0

    for _ in range(int(z_steps)):
        opt.zero_grad()

        pen = fusion_penalty(
            Z, float(lam), normalize=bool(normalize_fusion)
        )
        quad = 0.5 * float(rho) * torch.sum((Z - W) ** 2)
        obj = pen + quad

        if not torch.isfinite(obj):
            break

        obj.backward()
        opt.step()

        cur = float(obj.detach().item())
        if last is not None and abs(last - cur) <= float(early_tol) * (1.0 + abs(last)):
            stall += 1
            if stall >= int(early_patience):
                break
        else:
            stall = 0
        last = cur

    with torch.no_grad():
        Z2 = snap_rows(Z, float(snap_eps))
    return Z2.detach()
