#!/bin/bash
#SBATCH --job-name=bench_all
# ... (必要なSBATCH設定) ...

MODELS=("meta-llama/Meta-Llama-3.1-8B" "Qwen/Qwen2.5-7B" "mistralai/Mistral-7B-v0.3")
METHODS=("fp16" "turbo_quant" "rope_aware_tq" "ultra_quant")

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        # 評価スクリプトを連続実行
        python -m eval_pt.eval_ppl --model_name "$model" --method "$method"
        python -m eval_pt.eval_fidelity --model_name "$model" --method "$method"
        # NIAHなどは時間がかかるので条件付きにしても良い
    done
done