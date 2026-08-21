import argparse
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache

def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

def evaluate_leval(
    model,
    tokenizer,
    model_id,
    method,
    task_name,
    max_samples,
    device,
):
    print(f"=== L-Eval Eval [{task_name}] | Model: {model_id} | Method: {method} | Device: {device} ===")

    # L-Eval データセットのロード (Hugging Faceより)
    # タスクごとに構成が異なる場合があるため、汎用的な読み込み構造にしています
    try:
        dataset = load_dataset("L4NLP/LEval", task_name, split="test")
    except Exception as e:
        print(f"Error loading L-Eval dataset {task_name}: {e}")
        return {"status": "error", "message": str(e)}

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    predictions = []
    
    for item in tqdm(dataset, desc=f"Evaluating {task_name}"):
        # L-Eval は通常 'input' や 'instruction' を含む
        input_text = item.get("input", "") or item.get("instruction", "")
        
        # モデルへの入力プロンプト構築
        prompt = f"{input_text}\n\nAnswer:"
        
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=16000, truncation=True
        ).to(device)

        cache = QuantizedKVCache(method=method)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )

        gen_tokens = output[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        predictions.append(generated_text)

    return {
        "task_name": task_name,
        "evaluated_samples": len(predictions),
        "status": "success"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate L-Eval for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--task", type=str, default="finance", help="L-Eval task name (e.g., finance, law, science)")
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()

    target_device = get_optimal_device()
    torch_dtype = torch.float32 if target_device == "cpu" else torch.bfloat16

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype, device_map=target_device
    ).eval()

    metrics = evaluate_leval(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model_name,
        method=args.method,
        task_name=args.task,
        max_samples=args.max_samples,
        device=target_device
    )

    results_data = {
        "model_name": args.model_name,
        "method": args.method,
        "benchmark": f"leval_{args.task}",
        **metrics
    }

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_leval_{args.task}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_filename}")