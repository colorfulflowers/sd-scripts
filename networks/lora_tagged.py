# networks/lora_tagged.py
#
# =============================================================================
# 概要 / Overview
# =============================================================================
# sd-scripts の各モデル用 LoRA ネットワークモジュールをラップし、
# LoRA A パラメータに `_is_lora_A = True`、
# LoRA B パラメータに `_is_lora_B = True` 属性を自動付与するモジュールです。
#
# 主目的：spectral_normalization の正常動作
# -----------------------------------------------------------------------
# adv_optm の spectral_normalization は scale_update() 内で _is_lora_A タグを
# 参照して LoRA A と LoRA B を区別したスケーリングを行います。
# タグがない場合、A/B 両方が B 用スケーリングで処理されるため
# 学習が進まなくなります。
#
# 副作用：centered_wd の LoRA A/B への不適用
# -----------------------------------------------------------------------
# _is_lora_A / _is_lora_B タグが存在する場合、centered_decay.py の
# is_wd_centered() が True を返し、_init_anchor() が anchor_data を
# 作成しません。anchor_data が存在しない場合、apply_parameter_update() の
# centered_wd 処理条件（'anchor_data' in state）が満たされないため、
# centered_wd は LoRA A/B に対して機能しません。
#
# これは spectral_normalization と centered_wd（LoRA A/B）が
# 同じタグフラグ（_is_lora_A / _is_lora_B）を共有しているための制約です。
# centered_wd を LoRA A/B に対して有効にしたい場合は、
# このファイルを使用せず（タグなし）で学習してください。
# その場合 spectral_normalization は正しく機能しません。
#
# 参照 / References:
#   https://github.com/Koratahiu/Advanced_Optimizers/pull/28
#   https://github.com/Koratahiu/Advanced_Optimizers/pull/14
#
# =============================================================================
# 対応モデルとネットワークモジュール / Supported Models
# =============================================================================
#
#   モデル              学習スクリプト                   base_network_module
#   ----------------------------------------------------------------------------
#   SD1.x / SDXL / SD3 train_network.py             networks.lora       (デフォルト)
#   Flux.1              flux_train_network.py        networks.lora_flux
#   Anima               anima_train_network.py       networks.lora_anima
#   HunyuanImage-2.1    hunyuan_image_train_network  networks.lora_hunyuan_image
#   Lumina              lumina_train_network.py      networks.lora_lumina
#
# 全モデルで LoRA A = lora_down、LoRA B = lora_up の命名規則を使用しています。
#
# =============================================================================
# 使用方法 / Usage
# =============================================================================
#
# このファイルを sd-scripts の networks/ ディレクトリに配置します。
# 学習コマンドの --network_module を networks.lora_tagged に変更し、
# --network_args で base_network_module を指定します。
#
# --- SD1.x / SDXL / SD3 ---
#   --network_module networks.lora_tagged
#   --network_args "base_network_module=networks.lora"
#
# --- Flux.1 ---
#   --network_module networks.lora_tagged
#   --network_args "base_network_module=networks.lora_flux"
#
# --- Anima ---
#   --network_module networks.lora_tagged
#   --network_args "base_network_module=networks.lora_anima"
#
# --- HunyuanImage-2.1 ---
#   --network_module networks.lora_tagged
#   --network_args "base_network_module=networks.lora_hunyuan_image"
#
# --- Lumina ---
#   --network_module networks.lora_tagged
#   --network_args "base_network_module=networks.lora_lumina"
#
# base_network_module を省略した場合は networks.lora が使用されます。
#
# =============================================================================
# タグの効果まとめ / Effect of Tags
# =============================================================================
#
#   機能                     タグあり（本ファイル使用）  タグなし（本ファイル未使用）
#   -----------------------------------------------------------------------
#   spectral_normalization   正常動作 ✓                A/B 区別不可（誤動作）✗
#   centered_wd (LoRA A/B)   機能しない ✗              機能する（anchor=初期値）✓
#
#   spectral_normalization と centered_wd（LoRA A/B）は同時に正しく
#   機能させることができません。どちらを優先するかを選択してください。
#
# =============================================================================
# 注意事項 / Notes
# =============================================================================
#
# - タグ付けは apply_to() の完了後に実行されます。
#   sd-scripts の学習フロー上、オプティマイザ初期化（__init__ → init_step
#   → __init_state → _init_anchor）より前であることが保証されています。
#
# - チェックポイントから学習を再開する場合も同じ引数を指定してください。
#   タグ付けは毎回の学習開始時に自動で行われます。
#
# - sd-scripts が lora_down / lora_up 以外の命名規則を採用した場合は
#   _LORA_A_KEYWORD / _LORA_B_KEYWORD を更新してください。
#
# =============================================================================

import importlib
import sys

# LoRA A / LoRA B を識別するキーワード（全 sd-scripts モジュール共通）
_LORA_A_KEYWORD = "lora_down"
_LORA_B_KEYWORD = "lora_up"

# base_network_module 未指定時のデフォルト
_DEFAULT_BASE_MODULE = "networks.lora"


def _load_base_module(base_module_name: str):
    """
    指定されたベースモジュールをインポートして返す。
    インポート失敗時はユーザーへの案内付きエラーを発生させる。
    """
    try:
        return importlib.import_module(base_module_name)
    except ImportError as e:
        raise ImportError(
            f"[lora_tagged] base_network_module '{base_module_name}' のインポートに失敗しました: {e}\n"
            f"--network_args に正しい base_network_module を指定してください:\n"
            f"  SD1.x/SDXL/SD3:     base_network_module=networks.lora\n"
            f"  Flux.1:              base_network_module=networks.lora_flux\n"
            f"  Anima:               base_network_module=networks.lora_anima\n"
            f"  HunyuanImage-2.1:    base_network_module=networks.lora_hunyuan_image\n"
            f"  Lumina:              base_network_module=networks.lora_lumina"
        ) from e


def _tag_lora_params(network) -> tuple[int, int]:
    """
    network の全パラメータを走査し LoRA A / LoRA B にタグを付与する。

    タグの効果:
      _is_lora_A=True: spectral_normalization の scale_update() で A 専用スケーリングが適用される。
                       centered_decay.py の is_wd_centered() が True を返し
                       anchor_data が作成されないため centered_wd は機能しない。
      _is_lora_B=True: spectral_normalization で B 側として正しく処理される。
                       同様に centered_wd は機能しない。

    Returns:
        (tagged_A, tagged_B): タグ付けされた LoRA A / B のパラメータ数
    """
    tagged_a = 0
    tagged_b = 0
    for name, p in network.named_parameters():
        if _LORA_A_KEYWORD in name:
            p._is_lora_A = True
            tagged_a += 1
        elif _LORA_B_KEYWORD in name:
            p._is_lora_B = True
            tagged_b += 1
    return tagged_a, tagged_b


def _export_base_symbols(base_module) -> None:
    """
    ベースモジュールの公開シンボルをこのモジュールにエクスポートする。
    train_network.py 等が network_module から参照するシンボルを提供するために必要。
    create_network は lora_tagged 自身の実装を使うため除外する。
    """
    current_module = sys.modules[__name__]
    for attr_name in dir(base_module):
        if not attr_name.startswith("_") and attr_name != "create_network":
            if not hasattr(current_module, attr_name):
                setattr(current_module, attr_name, getattr(base_module, attr_name))


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    """
    ベースモジュールの create_network をラップし、
    apply_to() フック経由でタグ付けを自動化する。

    Args:
        base_network_module (str, optional):
            ラップするベースモジュール名。--network_args で指定。
            省略時は "networks.lora" が使用される。
        その他の引数はベースモジュールの create_network にそのまま渡される。
    """
    base_module_name = kwargs.pop("base_network_module", _DEFAULT_BASE_MODULE)
    base_module = _load_base_module(base_module_name)

    print(f"[lora_tagged] base_network_module: {base_module_name}")

    network = base_module.create_network(
        multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs
    )

    _export_base_symbols(base_module)

    # ------------------------------------------------------------------
    # apply_to をフックしてタグ付けを行う
    #
    # タイミングの根拠（sd-scripts の学習フロー）:
    #   1. create_network()            ← 本関数
    #   2. network.apply_to()          ← LoRA が model に適用 → params 確定
    #                                     ↑ ここでタグ付け実行
    #   3. prepare_optimizer_params()  ← タグ付き params をオプティマイザへ
    #   4. Optimizer.__init__()        ← init_step → __init_state
    #                                     → _init_anchor（タグ参照）
    #                                     → init_spectral_norm（タグ参照）
    #
    # apply_to 完了後（ステップ 2）にタグ付けすることで、
    # オプティマイザ初期化（ステップ 4）に確実に間に合う。
    # ------------------------------------------------------------------
    original_apply_to = network.apply_to

    def patched_apply_to(*args, **kwargs):
        result = original_apply_to(*args, **kwargs)

        tagged_a, tagged_b = _tag_lora_params(network)
        print(
            f"[lora_tagged] tagged {tagged_a} LoRA-A params (_is_lora_A=True), "
            f"{tagged_b} LoRA-B params (_is_lora_B=True)"
        )
        print(
            f"[lora_tagged] spectral_normalization: enabled (A/B distinction active)"
        )
        print(
            f"[lora_tagged] centered_wd: disabled for LoRA A/B "
            f"(anchor_data not created due to is_wd_centered()=True)"
        )

        if tagged_a == 0 and tagged_b == 0:
            print(
                f"[lora_tagged] WARNING: No params tagged. "
                f"Check that '{_LORA_A_KEYWORD}' / '{_LORA_B_KEYWORD}' "
                f"appear in parameter names of {base_module_name}."
            )

        return result

    network.apply_to = patched_apply_to
    return network
