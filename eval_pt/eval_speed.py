import argparse
import json
import os
import statistics
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache

"""
推論速度測定（量子化処理のオーバーヘッド測定）。

注意（主張の範囲）:
  本フレームワークはカーネル非改変の簡易実装であり、attention 自体は
  高精度（bf16）にデコードして行うため、低ビット KV キャッシュによる
  メモリ帯域削減効果は受けない。よって測定できるのは
  「量子化/逆量子化処理のオーバーヘッド込みの速度」だけであり、
  「量子化による高速化」を主張することはできない。
"""


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def measure_one_run(model, inputs, method, gen_tokens, device):
    """prefill + greedy decode(gen_tokens) の1回分を計測"""
    cache = QuantizedKVCache(method=method) if method != "fp16" else None

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            past_key_values=cache,
            use_cache=True
        )
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_time = time.time() - start

    past_key_values = outputs.past_key_values
    current_input_ids = inputs.input_ids[:, -1:]

    if device == "cuda":
        torch.cuda.synchronize()
    decode_start = time.time()
    for _ in range(gen_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        current_input_ids = next_token_id
        past_key_values = outputs.past_key_values
    if device == "cuda":
        torch.cuda.synchronize()
    decode_time = time.time() - decode_start

    return prefill_time, decode_time


def measure_inference_speed(model_name, method, seq_len=8192, gen_tokens=32, warmup_runs=1, measure_runs=3):
    print(f"=== Measuring Inference Speed | Model: {model_name} | Method: {method} | SeqLen: {seq_len} ===")

    device = get_optimal_device()
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    ).eval()

    dummy_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
    input_token_count = inputs.input_ids.shape[1]

    # ウォームアップ（JIT・キャッシュアロケータなどの初回コストを除外）
    for _ in range(warmup_runs):
        measure_one_run(model, inputs, method, gen_tokens, device)
        if device == "cuda":
            torch.cuda.empty_cache()

    prefills = []
    decodes = []
    for _ in range(measure_runs):
        p, d = measure_one_run(model, inputs, method, gen_tokens, device)
        prefills.append(p)
        decodes.append(d)
        if device == "cuda":
            torch.cuda.empty_cache()

    prefill_median = statistics.median(prefills)
    decode_median = statistics.median(decodes)
    tokens_per_second = gen_tokens / decode_median if decode_median > 0 else 0.0

    result = {
        "model_name": model_name,
        "method": method,
        "sequence_length": input_token_count,
        "prefill_time_sec": round(prefill_median, 4),
        "decode_time_sec": round(decode_median, 4),
        "generated_tokens": gen_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "all_prefill_runs": [round(x, 4) for x in prefills],
        "all_decode_runs": [round(x, 4) for x in decodes],
        "note": "中央値。カーネル非改変のため量子化オーバーヘッド込みの速度であり、高速化の主張はできない。",
    }

    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure KV Cache Inference Speed (median of runs)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--gen_tokens", type=int, default=32)
    parser.add_argument("--measure_runs", type=int, default=3)
    args = parser.parse_args()

    metrics = measure_inference_speed(
        model_name=args.model_name,
        method=args.method,
        seq_len=args.seq_len,
        gen_tokens=args.gen_tokens,
        measure_runs=args.measure_runs,
    )

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_speed.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Speed results successfully saved to {output_filename}")
