import argparse
import json
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache

"""
Needle In A Haystack（複数深度・複数試行の成功率評価）。

旧版の問題点:
  - 1 サンプル・1 深度（50%）だけで True/False を判定していたため統計的に検定力がなく、
    さらに当時のキャッシュバグ（decode が全履歴を参照しない）で全手法 False になった。

本バージョン:
  - 深度 {10%, 30%, 50%, 70%, 90%} × trials_per_depth 回で成功率 (success rate) を報告する。
  - 各試行で異なる秘密鍵を埋め込み、exact match（応答に鍵が含まれるか）で採点する。
"""


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def build_prompt(tokenizer, context_len, depth, secret_key, seed):
    """haystack 中の depth 位置に秘密鍵文を挿入したプロンプトを作る"""
    rng = random.Random(seed)
    haystack_base = "The history of modern computing dates back to the mid-20th century. "
    tokens_per_phrase = len(tokenizer.encode(haystack_base, add_special_tokens=False))
    repeat_count = max(1, (context_len - 100) // tokens_per_phrase)

    insert_pos = min(repeat_count - 1, max(0, int(repeat_count * depth)))
    needle = f" The secret key for HPC cluster access is {secret_key}. "

    prompt = (
        (haystack_base * insert_pos)
        + needle
        + (haystack_base * (repeat_count - insert_pos))
    )
    prompt += "\nWhat is the secret key for HPC cluster access? Answer:"
    _ = rng  # seed は将来の拡張用に固定しておく
    return prompt


def run_single_trial(model, tokenizer, prompt, cache, max_new_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    actual_len = inputs.input_ids.size(1)

    with torch.no_grad():
        position_ids = torch.arange(0, actual_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model(
            input_ids=inputs.input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    curr_pos = actual_len

    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        with torch.no_grad():
            pos_id = torch.tensor([[curr_pos]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=next_token,
                position_ids=pos_id,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
            curr_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_niah(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="turbo_quant",
    context_len=8192,
    depths=(0.1, 0.3, 0.5, 0.7, 0.9),
    trials_per_depth=1,
    max_new_tokens=16,
    seed=0,
    device=None,
):
    if device is None:
        device = get_optimal_device()

    print(
        f"=== Needle In A Haystack | Model: {model_id} | Context: {context_len} | "
        f"Method: {method} | Device: {device} ==="
    )

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

    trials = []
    successes = 0
    rng = random.Random(seed)

    for depth in depths:
        for trial in range(trials_per_depth):
            # 試行ごとに異なる鍵（各試行で推論に依存する値を使う）
            secret_key = rng.randint(100000000, 999999999)
            prompt = build_prompt(tokenizer, context_len, depth, secret_key, seed + trial)
            cache = QuantizedKVCache(method=method)

            response = run_single_trial(model, tokenizer, prompt, cache, max_new_tokens, device)
            ok = str(secret_key) in response
            successes += int(ok)
            trials.append({
                "depth": depth,
                "trial": trial,
                "secret_key": secret_key,
                "response": response,
                "success": ok,
            })
            print(f"  depth={depth:.0%} trial={trial} key={secret_key} success={ok}")

    n = len(trials)
    success_rate = successes / n if n > 0 else 0.0
    print(f"\n=== NIAH success rate: {successes}/{n} = {success_rate*100:.1f}% ===\n")
    return {
        "context_len": context_len,
        "depths": list(depths),
        "trials_per_depth": trials_per_depth,
        "n_trials": n,
        "n_success": successes,
        "niah_success_rate": success_rate,
        "trials": trials,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NIAH (multi-depth success rate) for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--context_len", type=int, default=8192)
    parser.add_argument("--trials_per_depth", type=int, default=1)
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    metrics = run_niah(
        model_id=args.model_name,
        method=args.method,
        context_len=args.context_len,
        trials_per_depth=args.trials_per_depth,
        device=target_device,
    )

    results_data = {
        "model_name": args.model_name,
        "method": args.method,
        **metrics
    }

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_niah.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_filename}")
