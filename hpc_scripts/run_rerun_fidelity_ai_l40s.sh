#!/bin/bash
#SBATCH --job-name=kv_quant_rerun_fidelity
#SBATCH --partition=ai-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/rerun_fidelity_%j.log
#SBATCH --error=logs/rerun_fidelity_%j.err

# 忠実度 (logit/KV fidelity) のみを再測定する。
# 目的: 旧 eval_fidelity.py は反復的な合成文を使っていたため Top-1 が飽和し
# (全条件が 1023/1024 = 99.90% 等の格子点に張り付く)、指標として識別力を
# 失っていた。修正版 (非反復の自然文) で全条件を再測定し、表の Top-1 列を
# 更新する。他の指標 (PPL/NIAH/LongBench/compression/speed) は飽和の影響を
# 受けないため再測定しない。
#
# 対象 (計10本):
#   - fidelity: 全5手法 (fp16 含む) x 両モデル, seq_len 1024
#     fp16 は fp16-vs-fp16 の回帰チェック兼用 (Top-1 が厳密に 100.00% になること)
#
# 出力ファイル名は既存と同じ ({model}_{method}_fidelity.json) のため、
# 成功したものから既存JSONが正規データで上書きされる。
# 条件 (seq_len 1024) は既存と同一。変えないこと。
#
# 完了後の後片付け (ローカルまたはクラスタ上で):
#   python results/summarize_results.py   # 表・プロットの再生成

cd /hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark

mkdir -p logs results/ai-l40s

echo "=================================================="
echo " Re-running Fidelity (natural non-repetitive text)"
echo " Working Directory: $(pwd)"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ai-l40s"
echo "=================================================="

# 上書き前に既存 results/ai-l40s を丸ごとバックアップ
BACKUP_DIR="results/ai-l40s_backup_$(date +%Y%m%d_%H%M%S)"
echo ">>> Backing up results/ai-l40s -> $BACKUP_DIR"
cp -r results/ai-l40s "$BACKUP_DIR"

MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
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
echo " Summary of re-measured Top-1 match rates"
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
