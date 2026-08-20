#!/bin/bash
#SBATCH --job-name=kv_bench_all
#SBATCH --output=logs/all_l40s_%j.log
#SBATCH --error=logs/all_l40s_%j.err
#SBATCH --partition=ng-dgx-m0
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# 環境変数のセットアップ
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Python 3.9 のバイナリを優先使用（無ければ通常の python3.9）
PY_BIN=$(which python3.9 2>/dev/null || echo "/usr/bin/python3.9")

echo "=================================================="
echo " Starting Full Evaluation Benchmark"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ng-dgx-m0"
echo " Python Bin: $PY_BIN"
echo "=================================================="

# 1. Perplexity (PPL) 評価
echo -e "\n[1/3] Running Perplexity Evaluation..."
$PY_BIN eval_pt/eval_ppl.py --model_name meta-llama/Meta-Llama-3.1-8B --method fp16 --seq_len 2048
$PY_BIN eval_pt/eval_ppl.py --model_name meta-llama/Meta-Llama-3.1-8B --method turbo_quant --seq_len 2048
$PY_BIN eval_pt/eval_ppl.py --model_name meta-llama/Meta-Llama-3.1-8B --method rope_aware_tq --seq_len 2048

# 2. Fidelity (CosSim / Top-k Match) 評価
echo -e "\n[2/3] Running Fidelity Evaluation..."
$PY_BIN eval_pt/eval_fidelity.py --model_name meta-llama/Meta-Llama-3.1-8B --method turbo_quant --seq_len 2048
$PY_BIN eval_pt/eval_fidelity.py --model_name meta-llama/Meta-Llama-3.1-8B --method rope_aware_tq --seq_len 2048

# 3. Needle In A Haystack (NIAH) 評価
echo -e "\n[3/3] Running Needle In A Haystack Evaluation..."
$PY_BIN eval_pt/eval_niah.py --model_name meta-llama/Meta-Llama-3.1-8B --method fp16 --context_len 8192
$PY_BIN eval_pt/eval_niah.py --model_name meta-llama/Meta-Llama-3.1-8B --method turbo_quant --context_len 8192
$PY_BIN eval_pt/eval_niah.py --model_name meta-llama/Meta-Llama-3.1-8B --method rope_aware_tq --context_len 8192

echo -e "\n=================================================="
echo " All Benchmark Evaluations Completed Successfully!"
echo " Date: $(date)"
echo "=================================================="