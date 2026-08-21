import argparse
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def get_optimal_device():
    """実行環境に合わせた最適なデバイスを自動判定"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def evaluate_zeroscrolls(
    model,
    tokenizer,
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="rope_aware_tq",
    task_name="gov_report",
    max_samples=20,
    device="cuda",
):
    print(f"=== ZeroSCROLLS Eval [{task_name}] | Model: {model_id} | Method: {method} | Device: {device} ===")

    # ZeroSCROLLS データセットのロード
    try:
        dataset = load_dataset("tau/zero_scrolls", task_name, split="validation")
    except Exception as e:
        print(f"Failed to load validation split, trying test split: {e}")
        dataset = load_dataset("tau/zero_scrolls", task_name, split="test")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    predictions = []
    references = []

    for item in tqdm(dataset, desc=f"Evaluating {task_name}"):
        input_text = item.get("input", "") or item.get("text", "")
        target_text = item.get("output", "") or item.get("summary", "")

        prompt = f"Document:\n{input_text}\n\nSummary/Answer:"
        
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=4096, truncation=True
        ).to(device)

        # 各サンプルの推論ごとに QuantizedKVCache を適用
        cache = QuantizedKVCache(method=method)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=128,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )

        gen_tokens = output[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        predictions.append(generated_text)
        references.append(target_text)

    print(f"Completed {len(predictions)} samples for task {task_name}.\n")
    
    return {
        "task_name": task_name,
        "evaluated_samples": len(predictions),
        "status": "success"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ZeroSCROLLS for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Hugging Face model ID")
    parser.add_argument("--method", type=str, default="turbo_quant", help="Quantization method")
    parser.add_argument("--task", type=str, default="gov_report", help="ZeroSCROLLS task name (e.g., gov_report, qmsum)")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to evaluate for quick test")
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    # 他の評価スクリプトと同様に、メイン部でモデルとトークナイザーをロード
    torch_dtype = torch.float32 if target_device == "cpu" else torch.bfloat16

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype, device_map=target_device
    ).eval()
    print("Model loaded successfully.\n")

    metrics = evaluate_zeroscrolls(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model_name,
        method=args.method,
        task_name=args.task,
        max_samples=args.max_samples,
        device=target_device
    )

    # 他のスクリプトと完全に統一された JSON 保存処理
    results_data = {
        "model_name": args.model_name,
        "method": args.method,
        "benchmark": f"zeroscrolls_{args.task}",
        **metrics
    }

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_zeroscrolls_{args.task}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_filename}")