# kvq-bench 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

[English](README.md) | **日本語**

大規模言語モデル（LLM）の **KVキャッシュ量子化技術** を比較するための、軽量でスタンドアロンなC++マイクロベンチマークフレームワークです。

2026年の最新文献に基づく、回転行列を用いる手法・レート歪み最適化手法・ハードウェアネイティブ手法などを比較します：**TurboQuant**、**RotorQuant**、**HyperQuant**、**UltraQuant**。

---

## 🌟 比較対象の手法

| 手法 | 主なアイデア | 回転オーバーヘッド | 対象ビット幅 |
| :--- | :--- | :---: | :---: |
| **TurboQuant** | デカルト座標→極座標変換（PolarQuant）＋ QJLによる1ビット符号補正 | 高（$O(d^2)$） | 3-bit / 8-bit |
| **RotorQuant** | Clifford代数（Cl(3,0)）のロータでフル直交回転を置き換え | 低（$O(d)$） | 3-bit / 8-bit |
| **HyperQuant** | RHT＋格子量子化（$A_2$/$D_4$/$E_8$）＋Riceエントロピー符号化による統一パイプライン | 中 | 1.7〜2 bps |
| **UltraQuant** | Walsh-Hadamard回転、QJL除去、ネイティブFP4（E2M1）＋UE8M0ブロックスケーリング | 低 | 4-bit |

---

## 📚 Related Work（関連研究）

### TurboQuant（Google Research、ICLR 2026）
2段階の圧縮を行う手法です：
- **PolarQuant**：ベクトルをデカルト座標から極座標に変換し、半径（大きさ）と角度（方向）に分離。角度分布が予測可能なため、従来必要だったブロック単位の正規化をスキップ可能。
- **QJL補正**：Johnson-Lindenstrauss変換で高次元データを縮小しつつ距離関係を保持し、各値を1ビットの符号（+1/-1）に削減。

| 指標 | 値 |
| :--- | :--- |
| 圧縮率 | 3ビットで6倍メモリ削減 |
| 速度向上 | Attention計算が最大8倍高速（H100） |
| 精度劣化 | ゼロ（LongBench, RULER等で検証済み） |
| 検証モデル | Gemma, Mistral |

**独立検証**：vLLMチームによる包括的な第三者検証（2026年5月）では、本番規模のモデル（Llama-3.3-70B-Instruct、Qwen3-30B-A3B、MiniMax-M2.7）を対象に、長文検索（MRCR）と推論系ベンチマーク（AIME25, GPQA, MATH500, LiveCodeBench-v6）で評価が行われました。結果は原論文よりも複雑で、FP8 KVキャッシュ量子化がBF16と同等のスループットのまま2倍の容量を確保し精度劣化もほぼ無しだった一方、TurboQuantの`k8v4`バリアントはFP8に対してわずかな優位性しかなく、スループットは40〜52%低下しました。`4bit-nc`バリアントはメモリ逼迫時に有用な、最も実用的なTurboQuant構成と評価された一方、より積極的な`k3v4-nc`／`3bit-nc`バリアントは難易度の高い推論・コーディングタスクで最大約20ポイントの精度低下が見られ、本番環境ではFP8 KVキャッシュが推奨されるデフォルトとされました。

参考文献：[arXiv:2504.19874](https://arxiv.org/abs/2504.19874)・[vLLM Blog — A First Comprehensive Study of TurboQuant](https://vllm.ai/blog/2026-05-11-turboquant)

### RotorQuant（Scrya）
TurboQuantのコアアルゴリズムをClifford代数（幾何代数）のロータで再設計した手法です。TurboQuantが用いるd×dのランダム直交回転行列（d=128で16,384回の乗算加算）を、Clifford代数 Cl(3,0) のロータ $R = \exp(B/2)$ に置き換えます。ベクトルを3次元グループに分割し、各グループに4パラメータのロータでサンドイッチ積 $RvR̃$ を適用します。

- 乗算加算：16,384 → 約2,064回（7.9倍削減、RotorQuant論文Table 1より）
- パラメータ数：16,399 → 372（44倍削減、RotorQuant論文Table 1より）

| 指標 | 値 |
| :--- | :--- |
| Perplexity（WikiText-2、Llama 3.1 8B Instruct、10.3倍圧縮時） | `iso3`: 6.91 vs TurboQuant `turbo3`: 7.07 — 同じ圧縮率でより高品質 |
| デコード速度 vs TurboQuant | 28%高速（119 vs 93 tok/s、RTX 5090） |
| プレフィル速度 vs TurboQuant | 5.3倍高速（3,822 vs 722 tok/s、RTX 5090） |
| パラメータ数 | 44倍少ない（372 vs 16,399、RotorQuant論文Table 1より） |
| 検証モデル | Llama 3.1 8B Instruct（主要ベンチマーク）、Qwen2.5-3B（デコード速度およびPython/TritonでのPerplexityベンチマーク）、MiniMax-M2.7（アーキテクチャ互換性チェックのみ、フルベンチマークではない） |

**重要な補足**：上記の「TurboQuantを上回る」という主要な結果は、実はRotorQuant本体（Clifford代数）ではなく、同じブロック対角回転のアイデアに基づく2つの派生手法——**IsoQuant**（4次元クォータニオン回転）と**PlanarQuant**（2次元ギブンス回転）——によるものです。この2つは別の貢献者（ParaMind2025）によって開発されました。リポジトリ自身が公開しているQwen2.5-3BでのPython/TritonによるPerplexity比較では、素のRotorQuant（3-bitでPPL 12.22、4-bitでPPL 10.03）はIsoQuant（4-bitでPPL 9.03）やPlanarQuant（3-bitでPPL 10.12）よりもむしろ性能が劣っています。リポジトリ内でもRotorQuant自体は「Research（Triton）」ステータス、IsoQuant/PlanarQuantは「Production（llama.cpp）」ステータスと明確に区別されています。以前記載していた「CUDA: 10-19倍、Metal: 9-31倍」という速度向上や、特定のコサイン類似度の数値については、現行のリポジトリの記載では確認できなかったため削除しました。

参考文献：[github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)

### HyperQuant
TurboQuantのような「KVキャッシュ専用の局所的なテクニック」を包含・凌駕し、「モデルの重み（Weights）」と「KVキャッシュ」の両方を同時に圧縮する統一（Unified）量子化パイプラインです。レート・歪み理論（Rate-Distortion Theory）を突き詰め、1.7〜2 bps（ビット/パラメータ）という極小のビット数まで精度を落とさずに圧縮します。

- **格子（多次元）量子化**：各成分を独立して丸める「スカラー量子化」ではなく、2次元・4次元・8次元空間で点を隙間なく詰め込める幾何学構造（$A_2, D_4, E_8$格子）を採用。同じビット数でも量子化誤差が空間の性質として大きく小さくなります。
- **エントロピー符号化**：RHT（ランダム・アダマール変換）で分布をガウス分布に変換した後、格子の幾何学的制約を利用して余分なビットを落とし（Bit-stripping）、発生頻度の高い値に短いビットを割り当てる可変長のRice符号化を施します。理論上の限界値のわずか約0.01 bps手前という圧縮効率を実現。
- **適用範囲**：LLMの重みとKVキャッシュの両方、さらにText-to-Videoモデル（19BパラメータのLTX-2など）といったDiffusion Transformers（DiT）の圧縮にも適用可能です。

| 指標 | 値 |
| :--- | :--- |
| vs TurboQuant / OCTOPUS（KVキャッシュ） | 1.7 bps（ビット/スカラー）まで両手法を上回る |
| vs HIGGS（重み） | 3〜5 bpsのすべての動作点で上回る |
| 圧縮率（H100、4 bps、ほぼロスレス） | 重み約3.9倍、KVキャッシュ約3.79倍 |
| レート歪みの理論限界とのギャップ | 約0.01 bps以内 |
| 検証モデル | Llama-3.1-8B（KVキャッシュのみのパス、bf16重み）、LTX-2（19BパラメータのDiT動画モデル、フレーム単位のアーティファクトなし） |
| Tritonカーネル速度向上 | *公開されている論文本文中には記載なし* |

参考文献：[arXiv:2606.23406](https://arxiv.org/abs/2606.23406)

### UltraQuant: 4-bit KV Caching for Context-Heavy Agents
これまでの手法（TurboQuant、HyperQuant）が抱えていた「システム実装・ハードウェア実行におけるボトルネック」を極限まで解消するために開発された、システム・ハードウェア連携型の4-bit KVキャッシュ圧縮手法です。理論的に美しいアルゴリズムが持ち込むソフト処理（コードブック参照やQJL補正）の遅延を捨て、最新GPUのハードウェア機能（FP4/FP8命令）に完璧に適合させた実践派、という位置づけです。

TurboQuantは「4-bitで高精度」を達成した画期的なアルゴリズムですが、実際にGPU（vLLMなど）上で動かすとデコード速度（Time-To-First-Tokenなど）が低下する弱点がありました。原因は、精度を保つための「ルックアップテーブル（コードブック参照）」や「1-bit残差補正（QJL）」を復元する際のソフトウェアループ処理が重すぎたためです。UltraQuantは特に長文を扱い、対話が続くエージェント型ワークロード（Context-Heavy Agents）において、メモリ容量を削減しつつ実際の推論スループットも高速にすることを目的に開発されました。

- **Walsh-Hadamard回転**：TurboQuantと同様に、Keyベクトルの外れ値を回転行列によって均一なガウス分布に分散。
- **QJL（残差補正）の切り捨て**：ソフトでのデコードオーバーヘッドが大きすぎるためあえて排除。
- **ハードウェアネイティブなFP4 Micro-tensor（E2M1）形式**：独自コードブックの代わりに、最新GPUがハードウェアレベルでネイティブサポートするFP4（E2M1形式：1符号ビット、2指数ビット、1仮数ビット）のグリッド値に直接マッピング。
- **ブロック単位のスケーリング（UE8M0）**：32チャンネルごとに共通のスケール係数を持たせ、オフラインで最適化した単一の定数 $c = 0.156$（論文中のアブレーションでMSE最適であることが確認済み）を乗算するシンプルなスケーリングを採用。これによりルックアップテーブルのソフト解読を排除し、GPUの行列演算器（MFMA / Tensor Core命令）にFP4データを直接流し込んで演算できます。

| 指標 | 値 |
| :--- | :--- |
| ヘッドライン：TTFT vs FP8 KV（エージェント型ワークロード、後半ラウンド／全ラウンド） | 3.47倍高速 / 2.3倍高速 |
| ヘッドライン：出力スループット vs FP8 KV（エージェント型ワークロード） | 1.63倍 |
| BF16比スループット（標準的なサービング、並列度64） | 1.38倍（FP8 KVの1.37倍とほぼ同等・誤差1%以内）、かつFP8の半分のKVバイト数で実現 |
| BF16比の中央値TPOT（1トークンあたり出力時間） | 1.40倍（FP8 KV: 1.37倍、Ultra-TQ: 1.58倍、vLLM OSS TurboQuant: 5.56倍） |
| UE8M0スケーリング定数 | $c = 0.156$ — MSE最適であることを確認済み。論文のGPQAアブレーションではFP8ベースラインを+4.4ポイント上回る |
| ハードウェア | AMD Instinct MI355X（CDNA4）、TP=2、ネイティブscaled-MFMA命令 |
| 検証モデル | MiniMax-M2.5（スループット／レイテンシ計測）／Qwen3.5-A3B、MiniMax-M2.5、Qwen2.5-72Bを対象とした本番想定の精度マトリクス（GPQA-Diamond、LCB-128K、AIME25、MATH500） |
| 精度への影響（本番想定マトリクス） | MATH500では安定〜微増（+0.0〜+0.8ポイント）、GPQA-DiamondとLCB-128Kでは競争力あり。一方で**AIME25では顕著な精度低下**（Qwen3.5-A3Bで−13.3ポイント、MiniMax-M2.5で−10.0ポイント、Qwen2.5-72Bで−3.3ポイント）が見られ、著者ら自身が「一律にほぼロスレスというわけではなくベンチマーク依存」と明言している。全結果はBoundary-layer protection（最初と最後の各2つのAttentionレイヤーはBF16のまま保持）を適用した状態のもの |

参考文献：[arXiv:2606.20474](https://arxiv.org/abs/2606.20474)

---

## 🧪 実験方法

### 計算資源
| プラットフォーム | ベンダー |
| :--- | :--- |
| ai-l40s | NVIDIA（L40S） |

### 使用するLLM
- Meta-Llama 3.1 8B
- Mistral 7B v0.3

### 実験内容
- **各ハードウェアでの実行速度＆速度向上**：8Kコンテキストでのプレフィル時間やトークン生成速度（トークン毎秒）を計測。
  → `eval_pt/eval_speed.py`
- **圧縮率**：8Kコンテキストにおいて、FP16ベースライン（289MB）を基準に、2ビット・4ビット・8ビットで検証。実際のKVキャッシュのメモリサイズをバイト単位で抽出し、FP16ベースラインに対する圧縮率を算出。
  → `eval_pt/eval_compression.py`
- **精度劣化＆各種タスク評価**：
  - **LongBench**：多タスク評価 → `eval_pt/eval_longbench.py`
  - **Needle In A Haystack（NIAH）**：長文中の特定情報検索精度 → `eval_pt/eval_niah.py`
- **注意機構の忠実度**（コサイン類似度、Top-1/Top-5トークン一致率）：オリジナルのFP16と量子化後のアテンション出力を直接比較。
  → `eval_pt/eval_fidelity.py`
- **Perplexity**（文章の自然さ・難解度）：
  → `eval_pt/eval_ppl.py`

---

## 📊 ベンチマーク指標（マイクロベンチマーク）

上記のモデルレベルの評価に加えて、C++マイクロベンチマーク自体は以下を計測します：

1. **エンコード＆デコードレイテンシ（µs）**：高分解能ハードウェアタイマーを用いてキー・ベクトルごとに測定。
2. **コサイン類似度**：FP32ベースラインに対する再構成KVベクトルの方向精度。
3. **Attention Logit MAE**：実際の$Q \cdot K^T$アテンションスコアに対する平均絶対誤差。

---

## 🚀 クイックスタート

### 前提条件

* C++17準拠のコンパイラ（`g++` または `clang++`）
* CMake 3.14以上
* （任意）プロット用に `pandas` と `matplotlib` を含むPython 3.8以上

### ビルドと実行

```bash
# 1. リポジトリをクローン
git clone https://github.com/your-username/kvq-bench.git
cd kvq-bench

# 2. CMakeでビルド
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. マイクロベンチマークを実行
./build/bench_kv
```

`./build/bench_kv` を実行すると、ターミナルにリアルタイムの実行統計が出力され、`results.csv` ファイルが生成されます。

### 結果のプロット

```bash
python3 scripts/plot_results.py
```

これにより、レイテンシと類似度のトレードオフを表示する `benchmark_result.png` が生成されます。

---

## 📁 リポジトリ構成

```
kvq-bench/
├── core_cpp/                  # 1. C++ 低レイヤーマイクロベンチ
│   ├── include/
│   │   ├── turbo_quant.hpp
│   │   ├── rotor_quant.hpp    # Rotor (Clifford代数)
│   │   ├── ultra_quant.hpp    # FP4(E2M1) + WHT
│   │   └── lattice_quant.hpp  # HyperQuant (E8/D4 Lattice)
│   └── bench_main.cpp
│
├── core_pt/                   # 2. PyTorch / CUDA カスタムカーネル・キャッシュ
│   ├── quantizers/
│   │   ├── base.py
│   │   ├── turbo_quant.py
│   │   ├── hyper_quant.py     # RHT + Lattice + Rice Code
│   │   └── ultra_quant.py     # WHT + FP4 direct map
│   ├── custom_cache.py        # DynamicCache インターフェース
│   └── kernels/                # (任意) Triton/CUDA カーネル
│
├── eval_pt/                    # 3. 実験・自動評価スクリプト
│   ├── eval_speed.py           # 8Kコンテキストでのプレフィル時間 & トークン毎秒
│   ├── eval_compression.py     # KVキャッシュのメモリサイズ vs FP16ベースライン
│   ├── eval_longbench.py       # LongBench 評価
│   ├── eval_niah.py            # Needle In A Haystack
│   ├── eval_fidelity.py        # Attention 忠実度 (Cosine, Top-1/5) vs FP16
│   └── eval_ppl.py             # Perplexity
│
└── hpc_scripts/                # 4. スパコン (Slurm) ジョブスクリプト
    └── run_l40s_cluster.sbatch # ai-l40s (NVIDIA L40S) 向け分散評価
```

* **`core_cpp/`**：上記クイックスタートで紹介しているスタンドアロンのC++17マイクロベンチマーク本体（エンコード/デコードレイテンシ、コサイン類似度、Attention Logit MAEを計測）。ヘッダオンリーの各量子化手法実装と、実行エントリポイントの`bench_main.cpp`から構成されます。
* **`core_pt/`**：実際のモデルの生成ループ内で使えるよう`DynamicCache`風インターフェースに組み込んだPython/PyTorch実装。任意のTriton/CUDAカーネルも含みます。
* **`eval_pt/`**：上記「実験方法」セクションに対応する、モデルレベルの自動評価スクリプト群（速度、圧縮率、LongBench、NIAH、Attention忠実度、Perplexity）。
* **`hpc_scripts/`**：ai-l40s GPUクラスタでベンチマークを実行するためのジョブスクリプト（`run_l40s_cluster.sbatch`）。

---

## 📄 引用・参考文献

この研究でこのベンチマークが役立った場合は、以下の元論文を引用してください：

* TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
* RotorQuant: [github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)
* HyperQuant: [arXiv:2606.23406](https://arxiv.org/abs/2606.23406) — HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline (2026)
* UltraQuant: [arXiv:2606.20474](https://arxiv.org/abs/2606.20474) — UltraQuant: 4-bit KV Caching for Context-Heavy Agents (2026)

---

## 📜 ライセンス

このプロジェクトは [MITライセンス](LICENSE) の下で公開されています。
