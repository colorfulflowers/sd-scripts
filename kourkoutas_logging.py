import torch


def get_real_optimizer(optimizer, lr_scheduler):
    # scheduler優先
    if lr_scheduler is not None and hasattr(lr_scheduler, "optimizers"):
        opt = lr_scheduler.optimizers[-1]
    else:
        opt = optimizer

    # DeepSpeed wrapper対応
    if hasattr(opt, "optimizer"):
        opt = opt.optimizer

    return opt


def collect_kourkoutas_metrics(optimizer):
    """
    Kourkoutas β₂の挙動解析用データを収集
    """

    if optimizer is None or not hasattr(optimizer, "kourkoutas_helper"):
        return None

    k = optimizer.kourkoutas_helper

    if not hasattr(k, "layer_state") or not k.layer_state:
        return None

    group = optimizer.param_groups[0]

    beta2_min = group.get("beta2_min", None)
    beta2_max = group.get("betas", (None, None))[1]

    beta2_vals = []
    raw_vals = []

    for state in k.layer_state.values():

        beta2 = state.get("dynamic_beta2")
        ema = state.get("kourkoutas_r_ema")
        acc = state.get("sum_sq_accumulator")

        if beta2 is None or ema is None or acc is None:
            continue

        if isinstance(beta2, torch.Tensor):
            beta2 = beta2.mean().item()

        r_ema = ema.mean().item()
        grad_norm = acc.sqrt().mean().item()

        beta2_vals.append(beta2)

        if r_ema > 0:
            raw_vals.append(grad_norm / (r_ema + 1e-12))

    if not beta2_vals:
        return None

    b_min_obs = min(beta2_vals)
    b_max_obs = max(beta2_vals)

    result = {
        "beta2_min_obs": b_min_obs,
        "beta2_max_obs": b_max_obs,
    }

    if beta2_min is not None and beta2_max is not None:
        util = (b_max_obs - b_min_obs) / (beta2_max - beta2_min + 1e-12)
        result["beta2_utilization"] = util

    if raw_vals:
        result["raw_mean"] = sum(raw_vals) / len(raw_vals)
        result["raw_max"] = max(raw_vals)

    return result
