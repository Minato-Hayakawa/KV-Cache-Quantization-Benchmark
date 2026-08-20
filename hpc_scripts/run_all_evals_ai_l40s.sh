#!/bin/bash
#SBATCH --job-name=eval_l40s
#SBATCH --output=logs/all_l40s_%j.log
#SBATCH --error=logs/all_l40s_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=ai-l40s

mkdir -p logs
mkdir -p results/ai-l40s

# 公式モジュールをロード
module load system/ai-l40s

# パスを通す
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
export PYTHONPATH="$(pwd):$(pwd)/eval:$(pwd)/eval_pt:$PYTHONPATH"

echo "=================================================="
echo "NVIDIA [ai-l40s] | Starting Full Evaluations"
echo "=================================================="

# 1. Perplexity (PPL) の評価
echo "-> Running PPL Evaluation..."
python3 -m eval_pt.eval_ppl

# 2. Fidelity の評価
echo "-> Running Fidelity Evaluation..."
python3 -m eval_pt.eval_fidelity

# 3. Needle In A Haystack (Niah) の評価
echo "-> Running NIAH Evaluation..."
python3 -m eval_pt.eval_niah

# 4. LongBench の評価
echo "-> Running LongBench Evaluation..."
python3 -m eval_pt.eval_longbench

echo "=================================================="
echo "NVIDIA [ai-l40s] | All Evaluations Finished Successfully!"
echo "=================================================="