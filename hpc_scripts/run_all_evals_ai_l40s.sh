#!/bin/bash
#SBATCH --job-name=kv_quant_eval
#SBATCH --partition=ai-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_eval_%j.log
#SBATCH --error=logs/full_eval_%j.err

# 作業ディレクトリの設定
cd /hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark

# ログ保存用ディレクトリが存在しない場合に自動作成する安全策
mkdir -p logs

# 環境の有効化（必要に応じてコメントアウトを解除してください）
# source ~/.bashrc
# conda activate your_env_name

echo "=================================================="
echo " Starting Comprehensive KV Cache Quantization Benchmark"
echo " Working Directory: $(pwd)"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ai-l40s"
echo "=================================================="

# 評価するモデルのリスト
MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# 評価する手法のリスト（RotorQuantの並列化対応済み）
METHODS=(
    "fp16"
    "turbo_quant"
    "rotor_quant"
    "hyper_quant"
    "ultra_quant"
)

for MODEL in "${MODELS[@]}"; do
    echo "##################################################"
    echo " Evaluating Model: $MODEL"
    echo "##################################################"

    # 理論値ベースの圧縮率を一括計算（GPU不要・高速）
    echo ">>> Running Theoretical Compression Rate Calculation (All Methods)..."
    python eval_pt/theoretical_compression.py --model_name "$MODEL" --seq_len 8192

    for METHOD in "${METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo " Current Method: $METHOD | Model: $MODEL"
        echo "--------------------------------------------------"

        # 1. PPL評価
        echo ">>> Running PPL Evaluation..."
        python eval_pt/eval_ppl.py --model_name "$MODEL" --method "$METHOD"

        # 2. 忠実度 (Fidelity) 評価
        echo ">>> Running Fidelity Evaluation..."
        python eval_pt/eval_fidelity.py --model_name "$MODEL" --method "$METHOD"

        # 3. Needle In A Haystack (NIAH) 評価
        echo ">>> Running NIAH Evaluation..."
        python eval_pt/eval_niah.py --model_name "$MODEL" --method "$METHOD" --context_len 8192

        # 4. LongBench 評価
        echo ">>> Running LongBench Evaluation..."
        python eval_pt/eval_longbench.py --model_name "$MODEL" --method "$METHOD"

        # 5. 速度向上の測定
        echo ">>> Running Speed Measurement..."
        python eval_pt/eval_speed.py --model_name "$MODEL" --method "$METHOD" --seq_len 8192 --gen_tokens 32

    done
done

echo "=================================================="
echo " All Comprehensive Benchmarks Completed Successfully!"
echo " Date: $(date)"
echo "=================================================="