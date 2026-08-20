#!/bin/bash
#SBATCH --job-name=eval_mi300x
#SBATCH --output=logs/all_mi300x_%j.log
#SBATCH --error=logs/all_mi300x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=fs-mi300x

mkdir -p logs
mkdir -p results/fs-mi300x

# 公式モジュールをロード
module load system/fs-mi300x

# ユーザー領域にインストールしたPythonパッケージのパスを通す
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"

echo "=================================================="
echo "AMD [fs-mi300x] | Starting Full Evaluations"
echo "=================================================="

# 1. Perplexity (PPL) の評価
echo "-> Running PPL Evaluation..."
python3 -m eval_pt.eval_ppl --output_dir results/fs-mi300x/ppl

# 2. Fidelity の評価
echo "-> Running Fidelity Evaluation..."
python3 -m eval_pt.eval_fidelity --output_dir results/fs-mi300x/fidelity

# 3. Needle In A Haystack (Niah) の評価
echo "-> Running NIAH Evaluation..."
python3 -m eval_pt.eval_niah --output_dir results/fs-mi300x/niah

# 4. LongBench の評価
echo "-> Running LongBench Evaluation..."
python3 -m eval_pt.eval_longbench --output_dir results/fs-mi300x/longbench

echo "=================================================="
echo "AMD [fs-mi300x] | All Evaluations Finished Successfully!"
echo "=================================================="