#!/bin/bash
#SBATCH --job-name=kv_quant_rerun_niah
#SBATCH --partition=ai-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/rerun_niah_%j.log
#SBATCH --error=logs/rerun_niah_%j.err

# NIAH の試行数増量 (--trials_per_depth 1 -> 5) と fp16 LongBench のみを対象に再実行する。
# run_all_evals_ai_l40s.sh は使わないこと (NIAH が trials_per_depth=1 固定で、
# 全7評価を最初からやり直すため)。
#
# 対象 (計12本):
#   - NIAH: 全5手法 (fp16 含む) x 両モデル (10本)
#           context 8192, trials_per_depth 5 (深度5 x 5試行 = 25試行/方法)
#   - LongBench: fp16 x 両モデル (2本), num_samples 10
#
# 出力ファイル名は既存と同じ ({model}_{method}_niah.json / {model}_fp16_longbench.json)
# のため、成功したものから既存JSONが正規データで上書きされる。
# 条件 (context 8192, num_samples 10, 深度) は既存と同一。変えないこと。

cd /hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark

mkdir -p logs results/ai-l40s

echo "=================================================="
echo " Re-running NIAH (trials_per_depth=5) + fp16 LongBench"
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

# 1. NIAH (全5手法 x 両モデル、trials_per_depth=5、計10本)
for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo " Method: $METHOD | Model: $MODEL"
        echo "--------------------------------------------------"
        echo ">>> Running NIAH Evaluation (multi-depth, trials_per_depth=5)..."
        python eval_pt/eval_niah.py --model_name "$MODEL" --method "$METHOD" --context_len 8192 --trials_per_depth 5
    done
done

# 2. LongBench (fp16 x 両モデル、計2本)
for MODEL in "${MODELS[@]}"; do
    echo "--------------------------------------------------"
    echo " Method: fp16 | Model: $MODEL"
    echo "--------------------------------------------------"
    echo ">>> Running LongBench Evaluation (fp16 re-run)..."
    python eval_pt/eval_longbench.py --model_name "$MODEL" --method fp16 --num_samples 10
done

echo "=================================================="
echo " Re-run Completed"
echo " Date: $(date)"
echo "=================================================="
