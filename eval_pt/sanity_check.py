import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_cache import QuantizedKVCache


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def greedy_generate(model, tokenizer, prompt, cache, max_new_tokens, device):
    """評価スクリプトと同じ手動 greedy decode（prefill→1トークンずつdecode）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.size(1)

    with torch.no_grad():
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model(
            input_ids=inputs.input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    curr_pos = seq_len

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

    return generated


def run_sanity_check(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    max_new_tokens=16,
    prompt_tokens=256,
    device=None,
):
    """
    回帰テスト（量子化評価の前提条件）:

    1. passthrough（量子化なし疑似量子化器）でキャッシュ配管を使った生成が
       fp16 ベースラインと【トークン完全一致】すること。
       → キャッシュの蓄積・連結・全履歴返却ロジックの正当性を検証する。
       （旧バグ: update() が最新チャンクのみを返し、decode がコンテキストを
       参照しない状態になっていた。このテストがあれば即検出できた。）

    2. 高ビット幅（7bit）の量子化器で fp16 に近い生成が得られること。
       → 量子化往復の数式が壊れていないことを検証する。
       （完全一致は要求しない。bf16 の丸めで分岐し得るため一致率を報告。）
    """
    if device is None:
        device = get_optimal_device()

    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"=== Sanity Check | Model: {model_id} | Device: {device} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
        )
        .eval()
    )

    prompt = (
        "The history of modern computing dates back to the mid-20th century. "
        "Quantum computing leverages superposition and entanglement. "
    ) * 12
    prompt += "Question: What does quantum computing leverage? Answer:"

    # --- ベースライン（fp16） ---
    baseline_cache = QuantizedKVCache(method="fp16")
    baseline_ids = greedy_generate(model, tokenizer, prompt, baseline_cache, max_new_tokens, device)
    print(f"[fp16]      generated: {baseline_ids}")

    results = {}

    # --- 1. passthrough: 完全一致を要求 ---
    pt_cache = QuantizedKVCache(method="passthrough")
    pt_ids = greedy_generate(model, tokenizer, prompt, pt_cache, max_new_tokens, device)
    n = min(len(baseline_ids), len(pt_ids))
    match_rate = sum(a == b for a, b in zip(baseline_ids[:n], pt_ids[:n])) / max(1, n)
    pt_pass = pt_ids == baseline_ids
    results["passthrough"] = {
        "generated_ids": pt_ids,
        "identical_to_fp16": pt_pass,
        "token_match_rate": match_rate,
        "criterion": "identical_to_fp16 == True",
    }
    print(f"[passthrough] generated: {pt_ids}")
    print(f"[passthrough] identical: {pt_pass} (match {match_rate*100:.1f}%)")

    # --- 2. 高ビット幅（7bit）: 近ロスレスであること ---
    for method in ["turbo_quant", "rotor_quant", "hyper_quant"]:
        cache = QuantizedKVCache(method=method, num_bits=7)
        ids = greedy_generate(model, tokenizer, prompt, cache, max_new_tokens, device)
        n = min(len(baseline_ids), len(ids))
        rate = sum(a == b for a, b in zip(baseline_ids[:n], ids[:n])) / max(1, n)
        results[f"{method}_7bit"] = {
            "generated_ids": ids,
            "token_match_rate_vs_fp16": rate,
            "criterion": "token_match_rate >= 0.5 (near-lossless at 7bit)",
        }
        print(f"[{method} 7bit] generated: {ids}")
        print(f"[{method} 7bit] match rate: {rate*100:.1f}%")

    overall_pass = results["passthrough"]["identical_to_fp16"]
    print(f"\n=== Sanity Check Overall: {'PASS' if overall_pass else 'FAIL'} ===")
    return {"overall_pass": overall_pass, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for QuantizedKVCache plumbing and quantizer math")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    metrics = run_sanity_check(model_id=args.model_name, max_new_tokens=args.max_new_tokens)

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_sanity.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Sanity check results saved to {output_filename}")

    # passthrough 不一致は評価全体を無効化するため非ゼロ終了
    if not metrics["overall_pass"]:
        raise SystemExit(1)
