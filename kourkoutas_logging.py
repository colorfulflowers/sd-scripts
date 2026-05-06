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

    # ハイブリッドオプティマイザ（Muon_adv等）では adam_ プレフィックスが付く
    prefix = "adam_" if group.get("adam_kourkoutas_beta", False) else ""

    # パラメータはすべて group から取得する。
    # PyTorchの仕組み上、optimizer_args で渡した値もデフォルト値も
    # 最初から param_groups に格納されているため、group.get() で十分。
    tiny_spike = group.get(f"{prefix}tiny_spike", 1e-8)
    eps        = group.get(f"{prefix}eps",        1e-8)
    beta2_min  = group.get(f"{prefix}beta2_min",  None)
    beta2_max  = (group.get(f"{prefix}betas") or (None, None))[1]

    beta2_vals = []
    raw_vals = []

    for state in k.layer_state.values():

        beta2 = state.get("dynamic_beta2")
        ema   = state.get("kourkoutas_r_ema")
        acc   = state.get("sum_sq_accumulator")

        if beta2 is None or ema is None or acc is None:
            continue

        if isinstance(beta2, torch.Tensor):
            beta2 = beta2.mean().item()

        r_ema     = ema.mean().item()
        grad_norm = acc.sqrt().mean().item()

        beta2_vals.append(beta2)

        if r_ema > 0:
            # tiny_spike は sunspike比の分母安定化用。論文実装に合わせて group から取得。
            raw_vals.append(grad_norm / (r_ema + tiny_spike))

    if not beta2_vals:
        return None

    b_min_obs = min(beta2_vals)
    b_max_obs = max(beta2_vals)

    result = {
        "beta2_min_obs": b_min_obs,
        "beta2_max_obs": b_max_obs,
    }

    if beta2_min is not None and beta2_max is not None:
        # beta2レンジがゼロになる場合の保護に eps を使用。
        util = (b_max_obs - b_min_obs) / (beta2_max - beta2_min + eps)
        result["beta2_utilization"] = util

    if raw_vals:
        result["raw_mean"] = sum(raw_vals) / len(raw_vals)
        result["raw_max"]  = max(raw_vals)

    return result
