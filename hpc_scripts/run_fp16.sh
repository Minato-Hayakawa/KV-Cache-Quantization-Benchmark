#!/bin/bash
#SBATCH --job-name=eval_fp16_base
#SBATCH --output=logs/fp16_baseline_%j.log
#SBATCH --error=logs/fp16_baseline_%j.err
#SBATCH --partition=l40s          # ご自身の環境のパーティション名に合わせてください
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

echo "=================================================="
echo "Starting FP16 Baseline PPL Evaluation (Llama-3.1-8B)"
echo "=================================================="

# 量子化なし（fp16）でPPL評価を実行
python eval_pt/eval_ppl.py --model_name "meta-llama/Meta-Llama-3.1-8B" --method fp16

echo "=================================================="
echo "FP16 Baseline Evaluation Finished!"
echo "=================================================="