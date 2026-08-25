import argparse
import json
import os
import string
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache

"""
LongBench 正式タスク（Qasper）による QA-F1 評価。

旧版の問題点:
  - 固定の擬似プロンプトを 1 本だけ生成させて「要約文を表示」するだけで、
    スコアリングが存在しなかった（スカラ指標ではなかった）。
  - GPU でも float16 を使っており、Llama 3.1 の長文でオーバーフローし得た。

本バージョン:
  - THUDM/LongBench の qasper（論文読解 QA）から先頭 N サンプルを取り、
    公式スタイルの QA-F1 で採点した平均値を報告する。
  - dtype は他の評価スクリプトと揃えて CUDA では bfloat16。
  - 「簡易ベンチマーク」として小さなサンプル数（デフォルト 10）に限定し、
    結果は方向性の観察として扱う（統計的厳密性は範囲外）。
"""


# --- LongBench 公式スタイルの QA-F1 スコアラ ---
def normalize_answer(s: str) -> str:
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def truncate_prompt_left(tokenizer, prompt, max_ctx_tokens, device):
    """長すぎるコンテキストは先頭側を切り捨てて質問部分を残す"""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    if ids.size(1) > max_ctx_tokens:
        ids = ids[:, -max_ctx_tokens:]
    return ids.to(device)


def run_longbench(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="turbo_quant",
    num_samples=10,
    max_ctx_tokens=6144,
    max_new_tokens=32,
    device=None,
):
    if device is None:
        device = get_optimal_device()

    print(f"=== LongBench (Qasper QA-F1) | Model: {model_id} | Method: {method} | Device: {device} ===")

    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
        )
        .eval()
    )

    from datasets import load_dataset
    try:
        dataset = load_dataset("THUDM/LongBench", "qasper", split="test")
    except Exception:
        # 新しいスキーマで失敗した場合は trust_remote_code を試す
        dataset = load_dataset("THUDM/LongBench", "qasper", split="test", trust_remote_code=True)

    num_samples = min(num_samples, len(dataset))

    scores = []
    sample_records = []
    for i in range(num_samples):
        sample = dataset[i]
        context = sample["context"]
        # LongBench のスキーマ: 質問文は "input" フィールド("question" キーは存在しない)
        question = sample["input"]
        answers = sample["answers"]  # 参照回答（複数）

        prompt = context + f"\n\nQuestion: {question}\nAnswer:"
        input_ids = truncate_prompt_left(tokenizer, prompt, max_ctx_tokens, device)

        cache = QuantizedKVCache(method=method)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        pred = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        score = max((qa_f1_score(pred, ans) for ans in answers), default=0.0)
        scores.append(score)
        sample_records.append({
            "index": i,
            "question": question[:200],
            "prediction": pred[:500],
            "references": answers[:3],
            "qa_f1": score,
        })
        print(f"  [{i}] F1={score:.3f} | pred: {pred[:80]!r}")

    mean_f1 = sum(scores) / len(scores) if scores else 0.0
    print(f"\n=== LongBench Qasper mean QA-F1 over {len(scores)} samples: {mean_f1:.4f} ===\n")

    return {
        "task": "qasper",
        "num_samples": len(scores),
        "max_ctx_tokens": max_ctx_tokens,
        "longbench_qasper_f1": mean_f1,
        "samples": sample_records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LongBench (Qasper QA-F1) for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_ctx_tokens", type=int, default=6144)
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    metrics = run_longbench(
        model_id=args.model_name,
        method=args.method,
        num_samples=args.num_samples,
        max_ctx_tokens=args.max_ctx_tokens,
        device=target_device,
    )

    results_data = {
        "model_name": args.model_name,
        "method": args.method,
        **metrics
    }

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_longbench.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_filename}")
