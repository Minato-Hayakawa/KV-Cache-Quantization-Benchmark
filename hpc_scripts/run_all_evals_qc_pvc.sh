#!/bin/bash
#SBATCH --job-name=eval_pvc
#SBATCH --output=logs/all_pvc_%j.log
#SBATCH --error=logs/all_pvc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=qc-pvc

mkdir -p logs
mkdir -p results/qc-pvc

# 公式モジュールをロード
module load system/qc-pvc

# ユーザー領域にインストールしたPythonパッケージのパスを通す
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"

echo "=================================================="
echo "Intel [qc-pvc] | Starting Full Evaluations"
echo "=================================================="

# 1. Perplexity (PPL) の評価
echo "-> Running PPL Evaluation..."
python3 -m eval_pt.eval_ppl --output_dir results/qc-pvc/ppl

# 2. Fidelity の評価
echo "-> Running Fidelity Evaluation..."
python3 -m eval_pt.eval_fidelity --output_dir results/qc-pvc/fidelity

# 3. Needle In A Haystack (Niah) の評価
echo "-> Running NIAH Evaluation..."
python3 -m eval_pt.eval_niah --output_dir results/qc-pvc/niah

# 4. LongBench の評価
echo "-> Running LongBench Evaluation..."
python3 -m eval_pt.eval_longbench --output_dir results/qc-pvc/longbench

echo "=================================================="
echo "Intel [qc-pvc] | All Evaluations Finished Successfully!"
echo "=================================================="