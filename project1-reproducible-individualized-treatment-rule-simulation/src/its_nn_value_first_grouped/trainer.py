# its_nn/trainer.py
import torch
import torch.nn.functional as F


def eval_mse(model, X, A, y):
    """
    Evaluate prediction MSE only.
    """
    model.eval()
    with torch.no_grad():
        y_hat = model(X, A).view(-1)
        y = y.view(-1)
        return float(F.mse_loss(y_hat, y).item())


def _prediction_loss(y_hat, y, loss_type="mse", huber_delta=1.0, target_scale: float = 1.0):
    loss_type = str(loss_type).lower()
    scale = max(float(target_scale), 1e-8)
    y_hat_s = y_hat / scale
    y_s = y / scale
    if loss_type == "huber":
        return F.huber_loss(y_hat_s, y_s, delta=float(huber_delta))
    elif loss_type == "mse":
        return F.mse_loss(y_hat_s, y_s)
    else:
        raise ValueError(f"Unknown loss_type={loss_type}")


def _tether_penalty(model, tether_Z):
    """
    Return raw tether penalty ||gamma - tether_Z||^2 / 2 form without mu.
    """
    gamma = model.gamma.weight
    return 0.5 * torch.sum((gamma - tether_Z.detach()) ** 2)


def _gamma_distance(model, tether_Z):
    """
    Frobenius distance ||gamma - tether_Z||_F.
    """
    gamma = model.gamma.weight
    return float(torch.norm(gamma - tether_Z.detach(), p="fro").item())


def _trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def _is_backbone_param_name(name: str) -> bool:
    """
    Heuristic for backbone parameters in Net:
      fc1, bn1, fc2, bn2, dropout
    """
    return (
        name.startswith("fc1.")
        or name.startswith("bn1.")
        or name.startswith("fc2.")
        or name.startswith("bn2.")
        or name.startswith("dropout.")
    )


def _is_gamma_param_name(name: str) -> bool:
    return name.startswith("gamma.")


def _is_baseline_param_name(name: str) -> bool:
    return name.startswith("m_head.")


def set_trainable_parts(
    model,
    train_backbone: bool = True,
    train_gamma: bool = True,
    train_baseline: bool = True,
):
    """
    Set requires_grad flags by module role.

    Returns:
        old_flags: dict[name -> old_requires_grad]
    """
    old_flags = {}

    for name, p in model.named_parameters():
        old_flags[name] = bool(p.requires_grad)

        if _is_backbone_param_name(name):
            p.requires_grad = bool(train_backbone)
        elif _is_gamma_param_name(name):
            p.requires_grad = bool(train_gamma)
        elif _is_baseline_param_name(name):
            p.requires_grad = bool(train_baseline)
        else:
            # Fallback: if some new parameter appears in future,
            # treat it as backbone-like by default.
            p.requires_grad = bool(train_backbone)

    return old_flags


def restore_trainable_parts(model, old_flags):
    """
    Restore requires_grad flags saved by set_trainable_parts.
    """
    if old_flags is None:
        return

    for name, p in model.named_parameters():
        if name in old_flags:
            p.requires_grad = bool(old_flags[name])


def summarize_trainable_parts(model):
    """
    Return a compact summary of trainable parameter counts by role.
    """
    n_backbone = 0
    n_gamma = 0
    n_baseline = 0
    n_other = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        n = p.numel()
        if _is_backbone_param_name(name):
            n_backbone += n
        elif _is_gamma_param_name(name):
            n_gamma += n
        elif _is_baseline_param_name(name):
            n_baseline += n
        else:
            n_other += n

    return {
        "backbone": n_backbone,
        "gamma": n_gamma,
        "baseline": n_baseline,
        "other": n_other,
        "total": n_backbone + n_gamma + n_baseline + n_other,
    }


def train_supervised(
    model,
    X_train, A_train, y_train,
    X_val,   A_val,   y_val,
    epochs: int,
    lr: float,
    wd: float,
    patience: int,
    eval_every: int,
    log_fn=print,
    tether_Z=None,
    mu_tether: float = 0.0,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    improve_tol: float = 1e-6,
    grad_clip: float = 5.0,
    target_scale: float = 1.0,
    # NEW: selective training control
    train_backbone: bool = True,
    train_gamma: bool = True,
    train_baseline: bool = True,
    restore_requires_grad: bool = True,
    evaluate_step0: bool = True,
):
    """
    Supervised training for outcome regression.

    Optimizes:
        pred_loss + mu_tether * 0.5 * ||gamma - tether_Z||^2

    New controls:
        train_backbone: whether to update feature extractor layers
        train_gamma: whether to update gamma embedding
        train_baseline: whether to update m_head
        restore_requires_grad: restore original requires_grad flags after training

    Returns:
        best_val_mse
    """
    old_flags = set_trainable_parts(
        model,
        train_backbone=train_backbone,
        train_gamma=train_gamma,
        train_baseline=train_baseline,
    )

    summary = summarize_trainable_parts(model)
    log_fn(
        "[Train] trainable parts | "
        f"backbone={summary['backbone']} | "
        f"gamma={summary['gamma']} | "
        f"baseline={summary['baseline']} | "
        f"other={summary['other']} | "
        f"total={summary['total']}"
    )

    params = _trainable_parameters(model)
    if len(params) == 0:
        if restore_requires_grad:
            restore_trainable_parts(model, old_flags)
        raise ValueError("No trainable parameters found. Check training-part flags.")

    opt = torch.optim.Adam(params, lr=float(lr), weight_decay=float(wd))

    best = float("inf")
    best_state = None
    bad = 0

    y_train = y_train.view(-1)
    y_val = y_val.view(-1)

    try:
        if evaluate_step0:
            model.eval()
            with torch.no_grad():
                initial_val_mse = eval_mse(model, X_val, A_val, y_val)
            best = float(initial_val_mse)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            log_fn(f"[Train] Step0 | val_mse={best:.6f}")

        for it in range(1, int(epochs) + 1):
            model.train()
            opt.zero_grad()

            y_hat = model(X_train, A_train).view(-1)

            pred_loss = _prediction_loss(
                y_hat, y_train,
                loss_type=loss_type,
                huber_delta=huber_delta,
                target_scale=target_scale,
            )

            tether_pen = torch.tensor(0.0, device=y_hat.device)
            gamma_dist = None

            if tether_Z is not None and float(mu_tether) > 0.0:
                tether_pen = _tether_penalty(model, tether_Z)
                total_loss = pred_loss + float(mu_tether) * tether_pen
                gamma_dist = _gamma_distance(model, tether_Z)
            else:
                total_loss = pred_loss

            total_loss.backward()

            if grad_clip is not None and float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))

            opt.step()

            if it % int(eval_every) == 0 or it == 1 or it == int(epochs):
                val_mse = eval_mse(model, X_val, A_val, y_val)

                msg = (
                    f"[Train] it={it:5d} | "
                    f"train_pred_loss={float(pred_loss.item()):.6f} | "
                    f"train_total_loss={float(total_loss.item()):.6f} | "
                    f"val_mse={val_mse:.6f}"
                )

                if tether_Z is not None and float(mu_tether) > 0.0:
                    msg += (
                        f" | tether_pen={float(tether_pen.item()):.6f}"
                        f" | gamma_dist={float(gamma_dist):.6f}"
                    )

                log_fn(msg)

                if val_mse < best - float(improve_tol):
                    best = val_mse
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    bad = 0
                    log_fn(f"[Train] New best at it={it:5d} | best_val_mse={best:.6f}")
                else:
                    bad += 1
                    if bad >= int(patience):
                        log_fn(
                            f"[Train] Early stop at it={it:5d} | "
                            f"best_val_mse={best:.6f} | "
                            f"no_improve_count={bad}"
                        )
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            log_fn("[Train] Warning: best_state is None, model kept at final iterate.")

    finally:
        if restore_requires_grad:
            restore_trainable_parts(model, old_flags)

    return best
