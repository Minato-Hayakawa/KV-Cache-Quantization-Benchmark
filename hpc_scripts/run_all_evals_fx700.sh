#!/bin/bash
#SBATCH --job-name=eval_fx700
#SBATCH --output=logs/all_fx700_%j.log
#SBATCH --error=logs/all_fx700_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=fx700

mkdir -p logs
mkdir -p results/fx700

# 公式モジュールをロード
module load system/fx700

# パスを通す
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
export PYTHONPATH="$(pwd):$(pwd)/eval:$(pwd)/eval_pt:$PYTHONPATH"

echo "=================================================="
echo "FX700 | Starting Full Evaluations"
echo "=================================================="

# 1. Perplexity (PPL) の評価
echo "-> Running PPL Evaluation..."
python3 -m eval_pt.eval_ppl --output_dir results/fx700/ppl

# 2. Fidelity の評価
echo "-> Running Fidelity Evaluation..."
python3 -m eval_pt.eval_fidelity --output_dir results/fx700/fidelity

# 3. Needle In A Haystack (Niah) の評価
echo "-> Running NIAH Evaluation..."
python3 -m eval_pt.eval_niah --output_dir results/fx700/niah

# 4. LongBench の評価
echo "-> Running LongBench Evaluation..."
python3 -m eval_pt.eval_longbench --output_dir results/fx700/longbench

echo "=================================================="
echo "FX700 | All Evaluations Finished Successfully!"
echo "=================================================="