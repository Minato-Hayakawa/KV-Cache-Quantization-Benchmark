#!/bin/bash
#SBATCH --job-name=eval_pvc
#SBATCH --output=logs/all_pvc_%j.log
#SBATCH --error=logs/all_pvc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --partition=qc-pvc

# 必要に応じたIntel環境の読み込み
# module load oneapi/latest

mkdir -p logs

MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "Qwen/Qwen2.5-7B"
    "mistralai/Mistral-7B-v0.3"
)

METHODS=(
    "fp16"
    "turbo_quant"
    "rope_aware_tq"
    "ultra_quant"
)

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "=================================================="
        echo "Intel XPU [qc-pvc] | Model: $model | Method: $method"
        echo "=================================================="

        python -m eval_pt.eval_ppl --model_name "$model" --method "$method"
        python -m eval_pt.eval_fidelity --model_name "$model" --method "$method"
        python -m eval_pt.eval_niah --model_name "$model" --method "$method"
        python -m eval_pt.eval_longbench --model_name "$model" --method "$method"

        echo ""
    done
done