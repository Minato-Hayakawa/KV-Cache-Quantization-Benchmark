# 実験結果サマリ（ai-l40s 再計測分）

`results/ai-l40s/*.json` から機械集計。対象プラットフォーム: ai-l40s (NVIDIA L40S)。
主なジョブ: 373124 + 373434（2026-08 全評価）、373457（NIAH を trials_per_depth=5 で全手法再実行 + fp16 LongBench 再実行）、373515（ultra_quant 専用 sanity）、373801（忠実度の再測定。※非反復の自然文に修正した eval_fidelity.py で実施。Top-1 飽和問題の修正分。ai-l40s 利用不可のため ai-h100l-pu（H100）で実行したが、数値結果は GPU 機種に非依存）。
fp16 LongBench は job 373457 で正規に再計測されたものに差し替え済み（復元ファイルは廃止。平均F1は旧ログ報告値と一致）。
NIAH は深度5 × 各5試行 = 計25試行に増量（job 373457）。集計データはリポジトリ直下の `summary_results.csv`（`python results/summarize_results.py` で生成）。

## 0. Sanity gate（回帰テスト）

| モデル | overall_pass | passthrough一致 | turbo 7bit | rotor 7bit | hyper 7bit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Meta-Llama-3.1-8B | True | True | 1.000 | 1.000 | 0.875 |
| Mistral-7B-Instruct-v0.3 | True | True | 1.000 | 1.000 | 1.000 |

注: ultra_quant は sanity_check.py の7bitテスト対象外（設計通り）。

## 1. KVキャッシュ footprint（8Kコンテキスト）

両モデルはKV形状が共通（32層 × 8 KVヘッド × 128 head_dim → fp16で 1,024 MB）のため一本の表。

| 手法 | designed_bits | stored_mb (int8実占有) | データ部 (MB) | メタデータ (MB) | designed footprint (MB) | fp16比圧縮率 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 1024.0 | 1024.0 | 0.0 | 1024.0 | 1.00倍 |
| turbo_quant | 3.0 | 528.0 | 192.0 | 16.0 | 208.0 | 4.92倍 |
| rotor_quant | 3.0 | 1208.0 | 192.0 | 704.0 | 896.0 | 1.14倍 |
| hyper_quant | 3.0 | 528.0 | 192.0 | 16.0 | 208.0 | 4.92倍 |
| ultra_quant | 4.0 | 528.0 | 256.0 | 16.0 | 272.0 | 3.76倍 |

## 2. Perplexity（WikiText-2、window 2048 / stride 512、低いほど良い）

| モデル | 手法 | PPL |
| :--- | :--- | ---: |
| Meta-Llama-3.1-8B | fp16 | 5.5667 |
| Meta-Llama-3.1-8B | turbo_quant | 6.8175 |
| Meta-Llama-3.1-8B | rotor_quant | 5.7484 |
| Meta-Llama-3.1-8B | hyper_quant | 7.3255 |
| Meta-Llama-3.1-8B | ultra_quant | 5.6922 |
| Mistral-7B-Instruct-v0.3 | fp16 | 4.8756 |
| Mistral-7B-Instruct-v0.3 | turbo_quant | 5.2196 |
| Mistral-7B-Instruct-v0.3 | rotor_quant | 4.9276 |
| Mistral-7B-Instruct-v0.3 | hyper_quant | 5.3571 |
| Mistral-7B-Instruct-v0.3 | ultra_quant | 4.9072 |

## 3. 忠実度（seq 1024、fp16との比較）

※2026-08-27 に非反復の自然文で再測定（job 373801、ai-h100l-pu）。旧測定（反復的な合成文）では Top-1 が全条件で 99.5〜99.90% に飽和していた（1023/1024≈99.90% は「1024 位置中 1 位置のみ argmax 反転」を意味する格子点で、反転は BOS 近傍の低マージン位置に集中）。自然文では不一致位置が系列全体に分散し、Top-1 は手法間を弁別する。他列も新テキストでの測定値に更新済み。

| モデル | 手法 | Logit cos | Logit Top-1 (%) | Logit Top-5重なり (%) | KV cos | KV 相対L2 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Meta-Llama-3.1-8B | fp16 | 1.000000 | 100.00 | 100.00 | 1.000000 | 0.0000 |
| Meta-Llama-3.1-8B | turbo_quant | 0.852395 | 91.80 | 57.13 | 0.852547 | 0.5203 |
| Meta-Llama-3.1-8B | rotor_quant | 0.987176 | 96.97 | 84.41 | 0.973811 | 0.2095 |
| Meta-Llama-3.1-8B | hyper_quant | 0.815868 | 91.31 | 56.07 | 0.830218 | 0.5629 |
| Meta-Llama-3.1-8B | ultra_quant | 0.989553 | 97.75 | 87.13 | 0.974999 | 0.2046 |
| Mistral-7B-Instruct-v0.3 | fp16 | 1.000000 | 100.00 | 100.00 | 1.000000 | 0.0000 |
| Mistral-7B-Instruct-v0.3 | turbo_quant | 0.985630 | 93.07 | 73.50 | 0.917008 | 0.3993 |
| Mistral-7B-Instruct-v0.3 | rotor_quant | 0.998638 | 97.85 | 93.59 | 0.986953 | 0.1559 |
| Mistral-7B-Instruct-v0.3 | hyper_quant | 0.983696 | 92.58 | 71.45 | 0.897866 | 0.4459 |
| Mistral-7B-Instruct-v0.3 | ultra_quant | 0.998957 | 97.66 | 93.89 | 0.988095 | 0.1485 |

## 4. NIAH（8Kコンテキスト、深度5 × 各5試行 = 計25試行の成功率、job 373457）

| モデル | 手法 | 成功率 | n_success / n_trials |
| :--- | :--- | :---: | :---: |
| Meta-Llama-3.1-8B | fp16 | 1.00 | 25 / 25 |
| Meta-Llama-3.1-8B | turbo_quant | 1.00 | 25 / 25 |
| Meta-Llama-3.1-8B | rotor_quant | 1.00 | 25 / 25 |
| Meta-Llama-3.1-8B | hyper_quant | 1.00 | 25 / 25 |
| Meta-Llama-3.1-8B | ultra_quant | 1.00 | 25 / 25 |
| Mistral-7B-Instruct-v0.3 | fp16 | 1.00 | 25 / 25 |
| Mistral-7B-Instruct-v0.3 | turbo_quant | 0.64 | 16 / 25 |
| Mistral-7B-Instruct-v0.3 | rotor_quant | 0.76 | 19 / 25 |
| Mistral-7B-Instruct-v0.3 | hyper_quant | 0.20 | 5 / 25 |
| Mistral-7B-Instruct-v0.3 | ultra_quant | 1.00 | 25 / 25 |

## 5. LongBench Qasper QA-F1（先頭10サンプル、左6144トークン切り捨て）

| モデル | 手法 | 平均 QA-F1 |
| :--- | :--- | ---: |
| Meta-Llama-3.1-8B | fp16 | 0.2997 |
| Meta-Llama-3.1-8B | turbo_quant | 0.1580 |
| Meta-Llama-3.1-8B | rotor_quant | 0.2551 |
| Meta-Llama-3.1-8B | hyper_quant | 0.1859 |
| Meta-Llama-3.1-8B | ultra_quant | 0.2832 |
| Mistral-7B-Instruct-v0.3 | fp16 | 0.3256 |
| Mistral-7B-Instruct-v0.3 | turbo_quant | 0.3135 |
| Mistral-7B-Instruct-v0.3 | rotor_quant | 0.2813 |
| Mistral-7B-Instruct-v0.3 | hyper_quant | 0.2582 |
| Mistral-7B-Instruct-v0.3 | ultra_quant | 0.2856 |

## 6. 速度（8Kコンテキスト、32トークン生成、3回中央値）

解釈は量子化オーバーヘッド込みの速度のみ。高速化の主張はできない。

| モデル | 手法 | prefill (s) | decode (s) | tokens/s | fp16比遅延倍率 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Meta-Llama-3.1-8B | fp16 | 1.2504 | 0.9545 | 33.52 | 1.00倍 |
| Meta-Llama-3.1-8B | turbo_quant | 1.2895 | 1.2064 | 26.53 | 1.26倍 |
| Meta-Llama-3.1-8B | rotor_quant | 1.4197 | 2.2416 | 14.28 | 2.35倍 |
| Meta-Llama-3.1-8B | hyper_quant | 1.3413 | 1.7750 | 18.03 | 1.86倍 |
| Meta-Llama-3.1-8B | ultra_quant | 1.6308 | 1.2326 | 25.96 | 1.29倍 |
| Mistral-7B-Instruct-v0.3 | fp16 | 1.2066 | 0.9167 | 34.91 | 1.00倍 |
| Mistral-7B-Instruct-v0.3 | turbo_quant | 1.2473 | 1.1662 | 27.44 | 1.27倍 |
| Mistral-7B-Instruct-v0.3 | rotor_quant | 1.3771 | 2.2435 | 14.26 | 2.45倍 |
| Mistral-7B-Instruct-v0.3 | hyper_quant | 1.2968 | 1.7350 | 18.44 | 1.89倍 |
| Mistral-7B-Instruct-v0.3 | ultra_quant | 1.5876 | 1.2001 | 26.67 | 1.31倍 |

