#!/bin/bash
#SBATCH --job-name=kv_quant_missing
#SBATCH --partition=ai-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/missing_eval_%j.log
#SBATCH --error=logs/missing_eval_%j.err

# 2026-08-26 の full_eval (job 373124) でクラッシュして欠損した評価のみを再実行する。
# 欠損内容:
#   - LongBench: 量子化4手法 x 両モデル (8本)  ... get_mask_sizes の4.5x/5.x非互換が原因
#   - Compression: 全5手法 x Mistral (5本)     ... config.head_dim が None でクラッシュ
#   - Analytical footprint: Mistral (1本)      ... 同上
# 成功済みの PPL / Fidelity / NIAH / Speed / SanityCheck / fp16 LongBench は再実行しない。

cd /hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark

mkdir -p logs results/ai-l40s

echo "=================================================="
echo " Re-running Missing Evaluations Only"
echo " Working Directory: $(pwd)"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ai-l40s"
echo "=================================================="

QUANT_METHODS=(
    "turbo_quant"
    "rotor_quant"
    "hyper_quant"
    "ultra_quant"
)

MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# 1. Analytical footprint (Mistral) ... GPU不要
echo ">>> Running Analytical Footprint Calculation... [mistralai/Mistral-7B-Instruct-v0.3]"
python eval_pt/theoretical_compression.py --model_name "mistralai/Mistral-7B-Instruct-v0.3" --seq_len 8192

# 2. Compression (Mistral, fp16 含む全5手法)
ALL_METHODS=(
    "fp16"
    "${QUANT_METHODS[@]}"
)
for METHOD in "${ALL_METHODS[@]}"; do
    echo ">>> Running Compression Evaluation... [mistralai/Mistral-7B-Instruct-v0.3 | $METHOD]"
    python eval_pt/eval_compression.py --model_name "mistralai/Mistral-7B-Instruct-v0.3" --method "$METHOD" --seq_len 8192
done

# 3. LongBench (量子化4手法 x 両モデル)
for MODEL in "${MODELS[@]}"; do
    for METHOD in "${QUANT_METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo " Method: $METHOD | Model: $MODEL"
        echo "--------------------------------------------------"
        echo ">>> Running LongBench Evaluation..."
        python eval_pt/eval_longbench.py --model_name "$MODEL" --method "$METHOD" --num_samples 10
    done
done

echo "=================================================="
echo " Missing Evaluations Completed"
echo " Date: $(date)"
echo "=================================================="
