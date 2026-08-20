#!/bin/bash
#SBATCH --job-name=kv_bench_all
#SBATCH --output=logs/all_l40s_%j.log
#SBATCH --error=logs/all_l40s_%j.err
#SBATCH --partition=qc-gh200        # ご自身の環境のパーティション名に合わせる
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# 1. 評価したいモデルのリストを配列で定義する
MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "Qwen/Qwen2.5-7B"
    "mistralai/Mistral-7B-v0.3"
)

echo "=================================================="
echo "NVIDIA [ai-l40s] | Starting Multi-Model Evaluations"
echo "=================================================="

# 2. モデルごとにループを回す
for model_id in "${MODELS[@]}"; do
    echo ""
    echo "##################################################"
    echo "Evaluating Model: $model_id"
    echo "##################################################"

    # PPL評価
    echo "-> Running PPL Evaluation..."
    python eval_pt/eval_ppl.py --model_name "$model_id" # ※スクリプトの引数仕様に合わせて調整してください

    # Fidelity評価
    echo "-> Running Fidelity Evaluation..."
    python eval_pt/eval_fidelity.py --model_name "$model_id"

    # NIAH評価
    echo "-> Running NIAH Evaluation..."
    python eval_pt/eval_niah.py --model_name "$model_id"

    echo "Finished evaluation for: $model_id"
    echo "--------------------------------------------------"
done

echo "=================================================="
echo "NVIDIA [ai-l40s] | All Models and Evaluations Finished!"
echo "=================================================="