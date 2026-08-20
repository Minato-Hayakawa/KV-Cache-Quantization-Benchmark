#!/bin/bash
#SBATCH --job-name=eval_mi300x
#SBATCH --output=logs/all_mi300x_%j.log
#SBATCH --error=logs/all_mi300x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=fs-mi300x

# 必要に応じたROCm環境の読み込み
# module load rocm/latest

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
        echo "AMD [fs-mi300x] | Model: $model | Method: $method"
        echo "=================================================="

        python -m eval_pt.eval_ppl --model_name "$model" --method "$method"
        python -m eval_pt.eval_fidelity --model_name "$model" --method "$method"
        python -m eval_pt.eval_niah --model_name "$model" --method "$method"
        python -m eval_pt.eval_longbench --model_name "$model" --method "$method"

        echo ""
    done
done