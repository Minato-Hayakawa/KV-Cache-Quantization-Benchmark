#!/bin/bash
#SBATCH --job-name=eval_h100
#SBATCH --output=logs/eval_h100_%j.log
#SBATCH --error=logs/eval_h100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=ai-h100l

mkdir -p logs
cd /path/to/KV-Cache-Quantization-Benchmark

# 必要に応じてモジュールや仮想環境を有効化
# module load cuda/12.2
# source /path/to/venv/bin/activate

python -m eval_pt.eval_ppl \
    --model_name "meta-llama/Llama-3-8B" \
    --method "turbo_quant" \
    --num_bits 3