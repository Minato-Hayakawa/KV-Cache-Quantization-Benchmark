#!/bin/bash
#SBATCH --job-name=eval_pvc
#SBATCH --output=logs/eval_pvc_%j.log
#SBATCH --error=logs/eval_pvc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=qc-pvc

mkdir -p logs
cd /path/to/KV-Cache-Quantization-Benchmark

# Intel oneAPI環境の読み込み例
# source /opt/intel/oneapi/setvars.sh

python -m eval_pt.eval_ppl \
    --model_name "meta-llama/Llama-3-8B" \
    --method "turbo_quant" \
    --num_bits 3