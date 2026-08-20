#!/bin/bash
#SBATCH --job-name=eval_fx700
#SBATCH --output=logs/eval_fx700_%j.log
#SBATCH --error=logs/eval_fx700_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        # A64FXのマルチコアを活かすため大きめに設定
#SBATCH --time=08:00:00           # CPU実行になるため長めに調整
#SBATCH --partition=fx700

mkdir -p logs
cd /path/to/KV-Cache-Quantization-Benchmark

# ARM/FX700向けの環境調整（必要に応じて）
# export OMP_NUM_THREADS=32

python -m eval_pt.eval_ppl \
    --model_name "meta-llama/Llama-3-8B" \
    --method "turbo_quant" \
    --num_bits 3