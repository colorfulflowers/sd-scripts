## Kourkoutas β₂ 挙動検証ログの追加
#### 目的

本変更の目的は、Prodigy_adv における kourkoutas_beta 有効時に、

EMA（kourkoutas_r_ema）の影響により β₂ が抑制されず、
設定された [beta2_min, beta2_max] のレンジを実際に使用できているか

を定量的に検証することである。

特に以下の挙動を確認対象とする：

β₂が理論レンジ全体に振幅しているか
EMAが過剰に働き、β₂が中央付近に固定されていないか
勾配変動（grad_norm）に対して適切に反応しているか

#### 背景

Kourkoutas β₂ は以下の式で決定される：

    raw = grad_norm / (r_ema + tiny_spike)
    sun = raw / (1 + raw)
    beta2 = beta2_max - (beta2_max - beta2_min) * sun

この構造により：

    raw ≪ 1 → β₂ ≈ beta2_max（安定）
    raw ≫ 1 → β₂ ≈ beta2_min（スパイク検出）

となる。

しかし、EMA（r_ema）が強すぎる場合：

    grad_norm ≈ r_ema
    → raw ≈ 1
    → β₂がレンジ中央に固定

となり、動的制御が実質無効化される。

#### 実装内容
1. TensorBoardログ拡張

train_network.py の generate_step_logs() に以下を追加：

取得データ：

    dynamic_beta2（layerごと）
    kourkoutas_r_ema
    sum_sq_accumulator（grad proxy）
    
2. 出力指標

以下のログを追加：

    k/beta2/min_obs
    k/beta2/max_obs
    k/beta2/utilization
    k/raw/mean
    k/raw/max

3. 最重要指標

    utilization = (max(beta2) - min(beta2)) / (beta2_max - beta2_min)

#### 評価基準

utilization	状態
    \< 0.2	EMAが強すぎ（β₂固定）
    0.2〜0.6	部分的に機能
    0.6〜0.9	正常
    > 0.9	理想（フルレンジ使用）

---

## バケット別詳細ログの追加（per_layer_logs）

#### 目的

`k/raw/mean` と `k/raw/max` はすべてのバケットを集計した値であり、
どのテンソル形状（バケット）でスパイクが発生しているかを特定できない。

本追加により、バケット別の `sunspike` 比率・β₂・勾配ノルム・EMAノルムを
TensorBoard 上で個別に観測できるようにする。

これにより `ema_alpha` や `beta2_min` の調整を、
全体統計ではなく実際にスパイクが起きているバケットに基づいて行えるようになる。

#### 実装内容

##### `kourkoutas_logging.py`

`collect_kourkoutas_metrics()` のループを `k.layer_state.items()` に変更し、
各バケットの詳細メトリクスをフラットな dict `per_layer_logs` として収集する。

GPU 同期コストの回避：  
`Kourkoutas.py` の `prepare_step()` が `layer_state` に保存する
`last_pooled_grad_norm` / `last_ema_norm`（CPU float）を優先して使用する。
これらが存在しない初回ステップのみ tensor から直接取得するフォールバックを使用する。

バケットキー：  
`layer_key_fn` のデフォルト（`tuple(p.shape)`）をそのまま文字列化して
TensorBoard タグ名に使用する。モデルが変わっても形状ベースのキーは
汎用的に機能するため、`layer_key_fn` の変更は不要。

##### `train_network.py`

既存の Kourkoutas ログブロック末尾に **1行のみ** 追加：

    logs.update(k_metrics.get("per_layer_logs", {}))

`per_layer_logs` が存在しない場合（Kourkoutas-β 無効時等）は
空の dict が返るため、既存動作への影響はない。

#### 追加される TensorBoard タグ

バケット1つあたり以下の4タグが追加される。

    k/layer/<shape>/sunspike   # grad_norm / (ema_norm + tiny_spike)
    k/layer/<shape>/beta2      # dynamic_beta2 の現在値
    k/layer/<shape>/grad_norm  # プールされた勾配ノルム
    k/layer/<shape>/ema_norm   # EMA ノルム（r_ema）

`<shape>` は例えば `(32, 320)`（rank=32, in_features=320 の LoRA down 行列）のような
テンソル形状の文字列。

LoRA（SD1.5 / SDXL）では典型的に 7 バケット × 4 タグ = 28 タグが追加される。
900 ステップ学習時の追加データ量は約 1.7 MB。

#### データ量見積もり

| 学習形式 | 追加バケット数 | 追加タグ数/step | 900step 追加量 |
|---|---|---|---|
| LoRA（SD1.5 / SDXL） | 〜7 | 28 | 約 1.7 MB |
| Full finetune SD1.5 | 〜20 | 80 | 約 4.8 MB |
| Full finetune SDXL | 〜30 | 120 | 約 7.2 MB |

#### 活用方法

`k/raw/max` が `k/raw/mean` を大きく上回るステップがある場合、
特定のバケットが sunspike を引き起こしている。
TensorBoard の `k/layer/*/sunspike` を並べて表示することで
どの形状のテンソルが支配的かを特定できる。

`ema_alpha` の調整は、sunspike が高いバケットを確認してから行うこと。
全体統計だけで調整すると、スパイクが均一でない場合に
スパイクしていないバケットまで影響を受ける。

# sd-scripts

[English](./README.md) / [日本語](./README-ja.md)

## 目次

<details>
<summary>クリックすると展開します</summary>

- [はじめに](#はじめに)
    - [スポンサー](#スポンサー)
    - [スポンサー募集のお知らせ](#スポンサー募集のお知らせ)
    - [更新履歴](#更新履歴)
    - [サポートモデル](#サポートモデル)
    - [機能](#機能)
- [ドキュメント](#ドキュメント)
    - [学習ドキュメント（英語および日本語）](#学習ドキュメント英語および日本語)
    - [その他のドキュメント](#その他のドキュメント)
    - [旧ドキュメント（日本語）](#旧ドキュメント日本語)
- [AIコーディングエージェントを使う開発者の方へ](#aiコーディングエージェントを使う開発者の方へ)
- [Windows環境でのインストール](#windows環境でのインストール)
    - [Windowsでの動作に必要なプログラム](#windowsでの動作に必要なプログラム)
    - [インストール手順](#インストール手順)
    - [requirements.txtとPyTorchについて](#requirementstxtとpytorchについて)
    - [xformersのインストール（オプション）](#xformersのインストールオプション)
- [Linux/WSL2環境でのインストール](#linuxwsl2環境でのインストール)
    - [DeepSpeedのインストール（実験的、LinuxまたはWSL2のみ）](#deepspeedのインストール実験的linuxまたはwsl2のみ)
- [アップグレード](#アップグレード)
    - [PyTorchのアップグレード](#pytorchのアップグレード)
- [謝意](#謝意)
- [ライセンス](#ライセンス)

</details>

## はじめに

Stable Diffusion等の画像生成モデルの学習、モデルによる画像生成、その他のスクリプトを入れたリポジトリです。

### スポンサー

このプロジェクトを支援してくださる企業・団体の皆様に深く感謝いたします。

<a href="https://aihub.co.jp/">
  <img src="./images/logo_aihub.png" alt="AiHUB株式会社" title="AiHUB株式会社" height="100px">
</a>

### スポンサー募集のお知らせ

このプロジェクトがお役に立ったなら、ご支援いただけると嬉しく思います。 [GitHub Sponsors](https://github.com/sponsors/kohya-ss/)で受け付けています。

### 更新履歴

- **Version 0.11.1 (2026-06-16):**
    - Anima LoRA／LLLite学習でtorch.compileサポートを追加しました。[PR #2379](https://github.com/kohya-ss/sd-scripts/pull/2379)
        - 学習が20%ほど高速化されるようです。動作にはTritonやMSVCコンパイラが必要です。詳細は[ドキュメント](./docs/anima_torch_compile.md)をご覧ください。
    - 2DのみのQwen-Image VAEを追加しました。[PR #2382](https://github.com/kohya-ss/sd-scripts/pull/2382)
        - [issue #2369](https://github.com/kohya-ss/sd-scripts/issues/2369) での woct0rdho 氏の提案に基づいています。woct0rdho 氏に感謝します。
        - `--qwen_image_vae_2d` を指定すると有効になります。重みは通常版（3D版）と同じものが使用できます。
        - latentの事前キャッシュの高速化が期待できます（学習自体は変わりません）。詳細は[ドキュメント](./docs/anima_train_network.md#memory-and-speed--メモリ速度関連)をご覧ください。
    - LLLiteインペインティングモデルの学習サポートを追加しました。[PR #2378](https://github.com/kohya-ss/sd-scripts/pull/2378)
        - 詳細は[ドキュメント](./docs/anima_train_control_net_lllite.md)をご覧ください。
    - timestep samplingの設定値のログ出力、timestepsの分布の可視化を追加しました。[PR #2384](https://github.com/kohya-ss/sd-scripts/pull/2384)
        - 可視化により学習がどのようなタイムステップで行われるかを理解しやすくなります。
        - 詳細は[ドキュメント](./docs/anima_train_network.md#visualizing-the-timestep-distribution)をご覧ください。

- **Version 0.11.0 (2026-06-12):**
    - コードベースの大規模な内部リファクタリングを行い、コードベースの品質と保守性を向上させました。[PR #2372](https://github.com/kohya-ss/sd-scripts/pull/2372)
        - ユーザーの方には直接の影響が極力少なくなるよう配慮しました。詳細について、および不具合報告などは[こちらのdiscussion](https://github.com/kohya-ss/sd-scripts/discussions/2358)までお願いします。

- **Version 0.10.6 (2026-06-12):**
    - リファクタリングマージ前の安定バージョン。

- **Version 0.10.5 (2026-05-08):**
    - transformersのバージョン5以降に対応しました。[PR #2315](https://github.com/kohya-ss/sd-scripts/pull/2315) および [PR #2316](https://github.com/kohya-ss/sd-scripts/pull/2316) marcus165090-spec氏に感謝します。
        - `requirements.txt`の`transformers`のバージョンは4.xのままですが、5.xでも動作します。何らかの理由で5.xを用いる場合はdiffusersもあわせて最新バージョンにしてください。
    - Anima向けのControlNet-LLLite学習に対応しました。[PR #2317](https://github.com/kohya-ss/sd-scripts/pull/2317)
        - 詳細は[ドキュメント](./docs/anima_train_control_net_lllite.md)をご覧ください。

- **Version 0.10.4 (2026-05-07):**
    - Intel GPUの互換性を向上しました。[PR #2307](https://github.com/kohya-ss/sd-scripts/pull/2307) WhitePr氏に感謝します。
    - SD 1.5/SDXLのinpaintingモデルの学習に対応しました。[PR #2309](https://github.com/kohya-ss/sd-scripts/pull/2309) および [PR #2318](https://github.com/kohya-ss/sd-scripts/pull/2318)allanoepping氏に感謝します。
        - 詳細は[ドキュメント](./docs/inpainting_training.md)をご覧ください。

### サポートモデル

* **Stable Diffusion 1.x/2.x**
* **SDXL**
* **SD3/SD3.5**
* **FLUX.1**
* **LUMINA**
* **HunyuanImage-2.1**
* **Anima**

### 機能

* LoRA学習
* fine-tuning（DreamBooth）：HunyuanImage-2.1以外のモデル
* Textual Inversion学習：SD/SDXL
* インペインティングモデル学習：SD1.5およびSDXL
* 画像生成
* その他、モデル変換やタグ付け、LoRAマージなどのユーティリティ

## ドキュメント

### 学習ドキュメント（英語および日本語）

日本語は折りたたまれているか、別のドキュメントにあります。

* [LoRA学習の概要](./docs/train_network.md)
* [データセット設定](./docs/config_README-ja.md) / [英語版](./docs/config_README-en.md)
* [高度な学習オプション](./docs/train_network_advanced.md)
* [SDXL学習](./docs/sdxl_train_network.md)
* [SD3学習](./docs/sd3_train_network.md)
* [FLUX.1学習](./docs/flux_train_network.md)
* [LUMINA学習](./docs/lumina_train_network.md)
* [HunyuanImage-2.1学習](./docs/hunyuan_image_train_network.md)
* [Fine-tuning](./docs/fine_tune.md)
* [Textual Inversion学習](./docs/train_textual_inversion.md)
* [ControlNet-LLLite学習](./docs/train_lllite_README-ja.md) / [英語版](./docs/train_lllite_README.md)
* [Anima向けControlNet-LLLite学習ガイド](./docs/anima_train_control_net_lllite.md)
* [Validation](./docs/validation.md)
* [マスク損失学習](./docs/masked_loss_README-ja.md) / [英語版](./docs/masked_loss_README.md)
* [インペインティング学習](./docs/inpainting_training.md)

### その他のドキュメント

* [画像生成スクリプト](./docs/gen_img_README-ja.md) / [英語版](./docs/gen_img_README.md)
* [WD14 Taggerによる画像タグ付け](./docs/wd14_tagger_README-ja.md) / [英語版](./docs/wd14_tagger_README-en.md)

### 旧ドキュメント（日本語）

* [学習について、共通編](./docs/train_README-ja.md) : データ整備やオプションなど
* [DreamBoothの学習について](./docs/train_db_README-ja.md)

## AIコーディングエージェントを使う開発者の方へ

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md` and/or `GEMINI.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt:

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  You can now add your own personal instructions below the import line (e.g., `Always respond in Japanese.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so they won't be committed to the repository.

このリポジトリでは、AIコーディングエージェント（例：Claude、Geminiなど）がプロジェクトのコンテキストやコーディング標準を理解できるようにするための推奨プロンプトを提供しています。

それらを使用するには、プロジェクトディレクトリに設定ファイルを作成して明示的に有効にする必要があります。

**簡単なセットアップ手順:**

1.  プロジェクトルートに `CLAUDE.md` や `GEMINI.md` ファイルを作成します。
2.  `CLAUDE.md` に以下の行を追加して、リポジトリの推奨プロンプトをインポートします。

    ```markdown
    @./.ai/claude.prompt.md
    ```

    またはGeminiの場合:

    ```markdown
    @./.ai/gemini.prompt.md
    ``` 
3.  インポート行の下に、独自の指示を追加できます（例：`常に日本語で応答してください。`）。

この方法により、エージェントに与える指示を各開発者が管理しつつ、リポジトリの推奨コンテキストを活用できます。`CLAUDE.md` および `GEMINI.md` は `.gitignore` に登録されているため、リポジトリにコミットされることはありません。

## Windows環境でのインストール

### Windowsでの動作に必要なプログラム

Python 3.10.xおよびGitが必要です。

- Python 3.10.x: https://www.python.org/downloads/windows/ からWindows installer (64-bit)をダウンロード
- git: https://git-scm.com/download/win から最新版をダウンロード

Python 3.11.x、3.12.xでも恐らく動作します（未テスト）。

PowerShellを使う場合、venvを使えるようにするためには以下の手順でセキュリティ設定を変更してください。
（venvに限らずスクリプトの実行が可能になりますので注意してください。）

- PowerShellを管理者として開きます。
- 「Set-ExecutionPolicy Unrestricted」と入力し、Yと答えます。
- 管理者のPowerShellを閉じます。

### インストール手順

PowerShellを使う場合、通常の（管理者ではない）PowerShellを開き以下を順に実行します。

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

コマンドプロンプトでも同一です。

（なお、python -m venv～の行で「python」とだけ表示された場合、py -m venv～のようにpythonをpyに変更してください。）

注：`bitsandbytes`、`prodigyopt`、`lion-pytorch` は `requirements.txt` に含まれています。

この例ではCUDA 12.4版をインストールします。異なるバージョンのCUDAを使用する場合は、適切なバージョンのPyTorchをインストールしてください。たとえばCUDA 12.1版の場合は `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121` としてください。

accelerate configの質問には以下のように答えてください。（bf16で学習する場合、最後の質問にはbf16と答えてください。）

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

※場合によって ``ValueError: fp16 mixed precision requires a GPU`` というエラーが出ることがあるようです。この場合、6番目の質問（
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``）に「0」と答えてください。（id `0`のGPUが使われます。）

### requirements.txtとPyTorchについて

PyTorchは環境によってバージョンが異なるため、requirements.txtには含まれていません。前述のインストール手順を参考に、環境に合わせてPyTorchをインストールしてください。

スクリプトはPyTorch 2.6.0でテストしています。PyTorch 2.6.0以降が必要です。

RTX 50シリーズGPUの場合、PyTorch 2.8.0とCUDA 12.8/12.9を使用してください。`requirements.txt`はこのバージョンでも動作します。

### xformersのインストール（オプション）

xformersをインストールするには、仮想環境を有効にした状態で以下のコマンドを実行してください。

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

必要に応じてCUDAバージョンを変更してください。一部のGPUアーキテクチャではxformersが利用できない場合があります。

## Linux/WSL2環境でのインストール

LinuxまたはWSL2環境でのインストール手順はWindows環境とほぼ同じです。`venv\Scripts\activate` の部分を `source venv/bin/activate` に変更してください。

※NVIDIAドライバやCUDAツールキットなどは事前にインストールしておいてください。

### DeepSpeedのインストール（実験的、LinuxまたはWSL2のみ）

DeepSpeedをインストールするには、仮想環境を有効にした状態で以下のコマンドを実行してください。

```bash
pip install deepspeed==0.16.7
```

## アップグレード

新しいリリースがあった場合、以下のコマンドで更新できます。

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

コマンドが成功すれば新しいバージョンが使用できます。

### PyTorchのアップグレード

PyTorchをアップグレードする場合は、[Windows環境でのインストール](#windows環境でのインストール)のセクションの`pip install`コマンドを参考にしてください。

## 謝意

LoRAの実装は[cloneofsimo氏のリポジトリ](https://github.com/cloneofsimo/lora)を基にしたものです。感謝申し上げます。

Conv2d 3x3への拡大は [cloneofsimo氏](https://github.com/cloneofsimo/lora) が最初にリリースし、KohakuBlueleaf氏が [LoCon](https://github.com/KohakuBlueleaf/LoCon) でその有効性を明らかにしたものです。KohakuBlueleaf氏に深く感謝します。

## ライセンス

スクリプトのライセンスはASL 2.0ですが（Diffusersおよびcloneofsimo氏のリポジトリ由来のものも同様）、一部他のライセンスのコードを含みます。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause
