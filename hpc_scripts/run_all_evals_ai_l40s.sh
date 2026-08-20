#!/bin/bash
#SBATCH --job-name=eval_l40s
#SBATCH --output=logs/all_l40s_%j.log
#SBATCH --error=logs/all_l40s_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=ai-l40s

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
        echo "NVIDIA [ai-l40s] | Model: $model | Method: $method"
        echo "=================================================="

        python -m eval_pt.eval_ppl --model_name "$model" --method "$method"
        python -m eval_pt.eval_fidelity --model_name "$model" --method "$method"
        python -m eval_pt.eval_niah --model_name "$model" --method "$method"
        python -m eval_pt.eval_longbench --model_name "$model" --method "$method"

        echo ""
    done
done