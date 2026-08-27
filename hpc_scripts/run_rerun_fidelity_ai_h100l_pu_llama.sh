#!/bin/bash
#SBATCH --job-name=kv_quant_rerun_fidelity_llama
#SBATCH --partition=ai-h100l-pu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00
#SBATCH --output=logs/rerun_fidelity_llama_%j.log
#SBATCH --error=logs/rerun_fidelity_llama_%j.err

# 忠実度 (logit/KV fidelity) のみを再測定する。【Llama-3.1-8B 側 / 分割 2本のうち 1本目】
#
# 目的: 旧 eval_fidelity.py は反復的な合成文を使っていたため Top-1 が飽和し
# (全条件が 1023/1024 = 99.90% 等の格子点に張り付く)、指標として識別力を
# 失っていた。修正版 (非反復の自然文) で全条件を再測定し、表の Top-1 列を
# 更新する。他の指標 (PPL/NIAH/LongBench/compression/speed) は飽和の影響を
# 受けないため再測定しない。
#
# 【なぜ分割か】
# ai-h100l-pu の最大実行時間は 30 分。一括版 (両モデル x 5手法 = 10本) は
# モデルロード (1本あたり数分) が支配的で 30 分に収まらなかったため、
# モデルごとに分割して 2 ジョブ並列で実行する (本スクリプト: Llama 側。
# もう一方: run_rerun_fidelity_ai_h100l_pu_mistral.sh)。
#
# 【計算資源についての特例】
# 本来の全評価は ai-l40s (NVIDIA L40S) パーティションで実施しているが、
# 再測定時点で ai-l40s が利用不可のため、【忠実度の再測定のみ】
# ai-h100l-pu (NVIDIA H100) パーティションを使用する。
# 量子化器の数値結果はデバイスに依存しないためパーティション変更の影響は
# ないが、速度値は本ジョブでは測定しないこと (速度表は ai-l40s の既存値)。
#
# 注意1: --gres=gpu:1 は【指定しない】こと。
#   ai-h100l-pu は専用パーティション ai-h100l とノード共用の全員向け
#   パーティション (最大実行時間30分) で、pu側にはGRESが公開されていない。
#   そのため --gres=gpu:1 を付けると投入時に
#     sbatch: error: Batch job submission failed: Requested node
#     configuration is not available
#   で弾かれる (実迷: 2026-08-27)。ノードには H100 NVL が1枚しかないため
#   GRES指定は不要。外せばジョブからGPUはそのまま見える。
#
# 注意2: -pu のノードでは GPU が MIG で 7 分割 (各 ~11GB) されており、
#   ジョブから見えるのは 10.75 GiB のスライスのみの場合がある。
#   その状態では 8B モデル (bf16 ~16GB) はロード時に CUDA OOM となる
#   (実迷: jobs 373811 / 373817 で全ラン OOM)。再実行が必要な場合は
#   MIG なしのノード/パーティションを使うこと。
#   ※ 本再測定の成果物 (全10件の fidelity JSON) は先行の一括ジョブ
#   373801 が MIG なしの状態で完走して出力済みであり、OOMだった
#   分割ジョブは既存JSONを一切上書きしていない (ロード時に落ちたため)。
#
# 対象 (計5本):
#   - fidelity: 全5手法 (fp16 含む) x Llama-3.1-8B, seq_len 1024
#     fp16 は fp16-vs-fp16 の回帰チェック兼用 (Top-1 が厳密に 100.00% になること)
#
# 出力ファイル名は既存と同じ ({model}_{method}_fidelity.json、
# results/ai-l40s/ 配下) のため、成功したものから既存JSONが正規データで
# 上書きされる。条件 (seq_len 1024) は既存と同一。変えないこと。
#
# 完了後の後片付け (ローカルまたはクラスタ上で):
#   python results/summarize_results.py   # 表・プロットの再生成

cd /hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark

mkdir -p logs results/ai-l40s

echo "=================================================="
echo " Re-running Fidelity (natural non-repetitive text)"
echo " Split job: Llama-3.1-8B only (1 of 2)"
echo " Working Directory: $(pwd)"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ai-h100l-pu (H100; exception for this fidelity"
echo "            re-run only, since ai-l40s was unavailable)"
echo "=================================================="

# GPU認識の確認 (GRESなしでGPUが見えているか。失敗時はここで止める)
nvidia-smi || { echo "!!! GPU not visible in job; aborting."; exit 1; }

# 上書き前に既存 results/ai-l40s を丸ごとバックアップ
BACKUP_DIR="results/ai-l40s_backup_$(date +%Y%m%d_%H%M%S)"
echo ">>> Backing up results/ai-l40s -> $BACKUP_DIR"
cp -r results/ai-l40s "$BACKUP_DIR"

MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
)

METHODS=(
    "fp16"
    "turbo_quant"
    "rotor_quant"
    "hyper_quant"
    "ultra_quant"
)

for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo " Method: $METHOD | Model: $MODEL"
        echo "--------------------------------------------------"
        echo ">>> Running Fidelity Evaluation (natural text, seq_len=1024)..."
        python eval_pt/eval_fidelity.py --model_name "$MODEL" --method "$METHOD" --seq_len 1024
    done
done

echo "=================================================="
echo " Summary of re-measured fidelity results"
echo "=================================================="
python - <<'EOF'
import glob, json, os
for path in sorted(glob.glob("results/ai-l40s/*_fidelity.json")):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    print(f"{j['model_name']:>40} | {j['method']:<12} | "
          f"Top-1={j['logit_top1_match_rate']:7.2f}% | "
          f"Top-5={j['logit_top5_overlap_rate']:7.2f}% | "
          f"logit cos={j['logit_cosine_similarity']:.4f} | "
          f"mismatch_pos_head={j.get('top1_mismatch_positions_head32', 'N/A (old format)')}")
EOF

echo "=================================================="
echo " Re-run Completed"
echo " Date: $(date)"
echo "=================================================="
