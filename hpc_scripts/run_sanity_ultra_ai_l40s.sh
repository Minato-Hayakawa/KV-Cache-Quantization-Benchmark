#!/bin/bash
#SBATCH --job-name=kv_quant_sanity_ultra
#SBATCH --partition=ai-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_ultra_%j.log
#SBATCH --error=logs/sanity_ultra_%j.err

# ultra_quant 専用 sanity 実験のみを実行する。
# run_all_evals_ai_l40s.sh には組み込まず、単独の回帰ゲートとして投げる用途。
# （本家 sanity_check.py の 7bit ループは ultra_quant を対象にできないための補完。
#   既存の評価パイプライン・結果ファイルには一切干渉しない。出力は
#   results/ai-l40s/{model}_sanity_ultra.json のみ）
#
# 内容:
#   検証A: HP bypass (FP4グリッド丸めのみ除外) で fp16 と greedy 16トークンの
#          一致率 >= 0.9 （回転・スケール・配管の数式検証）
#   検証B: ネイティブ4bit の compress→decompress 相対L2 < 0.40 かつ
#          全 index が [0,15] 内 （FP4グリッドマッピングの健全性）

cd /hs/work0/home/users/u0001988/KV-Cache-Quantization-Benchmark

mkdir -p logs results/ai-l40s

echo "=================================================="
echo " ultra_quant Dedicated Sanity Check"
echo " Working Directory: $(pwd)"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo " Partition: ai-l40s"
echo "=================================================="

MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

FAIL=0
for MODEL in "${MODELS[@]}"; do
    echo "--------------------------------------------------"
    echo " Model: $MODEL"
    echo "--------------------------------------------------"
    python eval_pt/sanity_check_ultra.py --model_name "$MODEL" || FAIL=1
done

echo "=================================================="
echo " ultra_quant Sanity Completed (FAIL=$FAIL)"
echo " Date: $(date)"
echo "=================================================="

exit $FAIL
