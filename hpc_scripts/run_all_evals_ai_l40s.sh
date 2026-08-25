#!/bin/bash
#SBATCH --job-name=kv_quant_eval
#SBATCH --partition=ai-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/full_eval_%j.log
#SBATCH --error=logs/full_eval_%j.err

# 作業ディレクトリの設定
cd /hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark

# ログ保存用ディレクトリが存在しない場合に自動作成する安全策
mkdir -p logs

echo "=================================================="
echo " Starting KV Cache Quantization Benchmark"
echo " Working Directory: $(pwd)"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ai-l40s"
echo "=================================================="

MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

METHODS=(
    "fp16"
    "turbo_quant"
    "rotor_quant"
    "hyper_quant"
    "ultra_quant"
)

for MODEL in "${MODELS[@]}"; do
    echo "##################################################"
    echo " Model: $MODEL"
    echo "##################################################"

    # 0. 回帰テスト（最重要：これが失敗したら以降の評価は無効）
    echo ">>> Running Sanity Check (passthrough + 7bit high-bit)..."
    if ! python eval_pt/sanity_check.py --model_name "$MODEL"; then
        echo "!!! Sanity check FAILED for $MODEL (cache plumbing is broken). Aborting."
        exit 1
    fi

    # 1. 解析的 footprint（GPU不要・実装の designed_bits から導出）
    echo ">>> Running Analytical Footprint Calculation..."
    python eval_pt/theoretical_compression.py --model_name "$MODEL" --seq_len 8192

    for METHOD in "${METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo " Method: $METHOD | Model: $MODEL"
        echo "--------------------------------------------------"

        # 2. 保存占有 + 解析的 footprint の実測
        echo ">>> Running Compression Evaluation..."
        python eval_pt/eval_compression.py --model_name "$MODEL" --method "$METHOD" --seq_len 8192

        # 3. PPL
        echo ">>> Running PPL Evaluation..."
        python eval_pt/eval_ppl.py --model_name "$MODEL" --method "$METHOD"

        # 4. 忠実度（logit + KV ベクトル）
        echo ">>> Running Fidelity Evaluation..."
        python eval_pt/eval_fidelity.py --model_name "$MODEL" --method "$METHOD" --seq_len 1024

        # 5. NIAH（複数深度の成功率）
        echo ">>> Running NIAH Evaluation (multi-depth)..."
        python eval_pt/eval_niah.py --model_name "$MODEL" --method "$METHOD" --context_len 8192 --trials_per_depth 1

        # 6. LongBench（Qasper QA-F1, 小サンプル）
        echo ">>> Running LongBench Evaluation..."
        python eval_pt/eval_longbench.py --model_name "$MODEL" --method "$METHOD" --num_samples 10

        # 7. 速度（中央値・量子化オーバーヘッドとして解釈）
        echo ">>> Running Speed Measurement..."
        python eval_pt/eval_speed.py --model_name "$MODEL" --method "$METHOD" --seq_len 8192 --gen_tokens 32 --measure_runs 3
    done
done

echo "=================================================="
echo " All Benchmarks Completed"
echo " Date: $(date)"
echo "=================================================="
