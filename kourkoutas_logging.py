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
    Kourkoutas β₂の挙動解析用データを収集。

    グローバル統計に加え、バケット別の詳細メトリクスを per_layer_logs として返す。
    per_layer_logs はそのまま logs.update() できるフラットな dict。

    バケットキーは layer_key_fn が返す tuple（デフォルトはテンソル形状）を
    文字列化したもの（例: "(32, 320)"）をタグ名に使用する。
    モデルが変わっても形状ベースのキーは汎用的に機能するため、
    layer_key_fn のデフォルト動作をそのまま利用する。

    各バケットの値は prepare_step() が保存した CPU float を優先して使用し、
    GPU 同期コストを発生させない。prepare_step() 前の初回ステップのみ
    tensor から直接取得するフォールバックを使用する。

    Returns:
        dict | None: 以下のキーを含む dict。
            グローバル統計（既存）:
                "beta2_min_obs"     : float
                "beta2_max_obs"     : float
                "beta2_utilization" : float
                "raw_mean"          : float
                "raw_max"           : float
            バケット別詳細（追加）:
                "per_layer_logs"    : dict  TensorBoard タグ → float のフラット dict
                    例:
                        "k/layer/(32, 320)/sunspike"  : float
                        "k/layer/(32, 320)/beta2"     : float
                        "k/layer/(32, 320)/grad_norm" : float
                        "k/layer/(32, 320)/ema_norm"  : float
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

    beta2_vals   = []
    raw_vals     = []
    per_layer_logs = {}

    for layer_key, state in k.layer_state.items():

        beta2 = state.get("dynamic_beta2")
        ema   = state.get("kourkoutas_r_ema")
        acc   = state.get("sum_sq_accumulator")

        if beta2 is None or ema is None or acc is None:
            continue

        if isinstance(beta2, torch.Tensor):
            beta2 = beta2.mean().item()

        # prepare_step() が保存した CPU float を優先（GPU 同期不要）。
        # 初回ステップ等でまだ保存されていない場合のみ tensor から取得。
        grad_norm_cpu = state.get("last_pooled_grad_norm")
        ema_norm_cpu  = state.get("last_ema_norm")

        if grad_norm_cpu is not None and ema_norm_cpu is not None:
            grad_norm = grad_norm_cpu
            r_ema     = ema_norm_cpu
        else:
            r_ema     = ema.mean().item()
            grad_norm = acc.sqrt().mean().item()

        beta2_vals.append(beta2)

        if r_ema > 0:
            # tiny_spike は sunspike比の分母安定化用。論文実装に合わせて group から取得。
            sunspike = grad_norm / (r_ema + tiny_spike)
            raw_vals.append(sunspike)
        else:
            sunspike = 0.0

        # バケット別ログ: タグ名に layer_key を文字列化して使用。
        # layer_key_fn のデフォルトは tuple(p.shape) なので
        # "(32, 320)" のようなモデル非依存のタグになる。
        tag = str(layer_key)
        per_layer_logs[f"k/layer/{tag}/sunspike"]  = sunspike
        per_layer_logs[f"k/layer/{tag}/beta2"]     = beta2
        per_layer_logs[f"k/layer/{tag}/grad_norm"] = grad_norm
        per_layer_logs[f"k/layer/{tag}/ema_norm"]  = r_ema

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

    if per_layer_logs:
        result["per_layer_logs"] = per_layer_logs

    return result
