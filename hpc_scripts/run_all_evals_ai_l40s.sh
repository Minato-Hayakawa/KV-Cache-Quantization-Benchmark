#!/bin/bash
#SBATCH --job-name=kv_bench_full
#SBATCH --output=logs/full_eval_%j.log
#SBATCH --error=logs/full_eval_%j.err
#SBATCH --partition=ai-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

export PYTHONUNBUFFERED=1

# プロジェクトのルートディレクトリに移動し、パスを通す
PROJECT_DIR="/hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark"
cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# 作成した仮想環境の Python の絶対パス
PY_BIN="$PROJECT_DIR/venv_l40s/bin/python"

# 実験計画に沿ったモデルと手法の定義
MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
METHODS=("fp16" "turbo_quant" "rotor_quant" "hyper_quant" "ultra_quant")

echo "=================================================="
echo " Starting Comprehensive KV Cache Quantization Benchmark"
echo " Working Directory: $(pwd)"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ai-l40s"
echo "=================================================="

for model in "${MODELS[@]}"; do
    echo "##################################################"
    echo " Evaluating Model: $model"
    echo "##################################################"

    for method in "${METHODS[@]}"; do
        echo "--------------------------------------------------"
        echo " Current Method: $method | Model: $model"
        echo "--------------------------------------------------"

        # 1. Perplexity (PPL) 評価
        echo ">>> Running PPL Evaluation..."
        $PY_BIN eval_pt/eval_ppl.py --model_name "$model" --method "$method" --seq_len 2048

        # 2. Fidelity (CosSim / Top-k Match) 評価
        echo ">>> Running Fidelity Evaluation..."
        $PY_BIN eval_pt/eval_fidelity.py --model_name "$model" --method "$method" --seq_len 2048

        # 3. Needle In A Haystack (NIAH) 評価
        echo ">>> Running NIAH Evaluation..."
        $PY_BIN eval_pt/eval_niah.py --model_name "$model" --method "$model" --context_len 8192

        # 4. LongBench 評価
        echo ">>> Running LongBench Evaluation..."
        $PY_BIN eval_pt/eval_longbench.py --model_name "$model" --method "$method"

        # 5. ZeroSCROLLS 評価 (例: gov_report)
        echo ">>> Running ZeroSCROLLS Evaluation..."
        $PY_BIN eval_pt/eval_zeroscrolls.py --model_name "$model" --method "$method" --task "gov_report" --max_samples 10

        # 6. L-Eval 評価 (例: finance)
        echo ">>> Running L-Eval Evaluation..."
        $PY_BIN eval_pt/eval_leval.py --model_name "$model" --method "$method" --task "finance" --max_samples 10
    done
done

echo -e "\n=================================================="
echo " All Comprehensive Benchmarks Completed Successfully!"
echo " Date: $(date)"
echo "=================================================="