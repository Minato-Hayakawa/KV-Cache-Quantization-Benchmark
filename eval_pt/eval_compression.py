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

def measure_vram_usage(model_name, method, seq_len=8192):
    print(f"=== Measuring VRAM Delta & Compression Rate | Model: {model_name} | Method: {method} | SeqLen: {seq_len} ===")
    
    device = get_optimal_device()
    if device != "cuda":
        raise RuntimeError("VRAM peak memory measurement requires a CUDA device (GPU).")

    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ダミーの長文コンテキストを生成
    dummy_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)

    # -------------------------------------------------------------
    # 1. FP16（ベースライン）でのKVキャッシュ追加メモリ量を測定
    # -------------------------------------------------------------
    print("--- Measuring FP16 Baseline KV Memory Delta ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    ).eval()

    # モデル重みロード直後のベースVRAMを記録
    base_model_mem = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        _ = model(
            input_ids=inputs.input_ids,
            past_key_values=None,
            use_cache=True
        )

    fp16_peak_mem = torch.cuda.max_memory_allocated(device)
    # KVキャッシュによって増加した純粋なメモリ量（バイト）
    fp16_kv_bytes = max(0, fp16_peak_mem - base_model_mem)
    fp16_kv_mb = fp16_kv_bytes / (1024 * 1024)
    print(f"FP16 Baseline KV Memory Delta: {fp16_kv_mb:.2f} MB")

    # クリーンアップ
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # -------------------------------------------------------------
    # 2. 指定された量子化手法（method）でのKVキャッシュ追加メモリ量を測定
    # -------------------------------------------------------------
    print(f"--- Measuring {method} KV Memory Delta ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    ).eval()

    # 量子化モデルロード直後のベースVRAMを記録
    quant_base_model_mem = torch.cuda.memory_allocated(device)

    cache = QuantizedKVCache(method=method) if method != "fp16" else None

    with torch.no_grad():
        _ = model(
            input_ids=inputs.input_ids,
            past_key_values=cache,
            use_cache=True
        )

    quant_peak_mem = torch.cuda.max_memory_allocated(device)
    # 量子化KVキャッシュによって増加した純粋なメモリ量（バイト）
    quant_kv_bytes = max(0, quant_peak_mem - quant_base_model_mem)
    quant_kv_mb = quant_kv_bytes / (1024 * 1024)
    print(f"{method} KV Memory Delta: {quant_kv_mb:.2f} MB")

    # クリーンアップ
    del model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. 圧縮率の計算（FP16の増加量 / 量子化手法の増加量）
    # -------------------------------------------------------------
    compression_ratio = fp16_kv_mb / quant_kv_mb if quant_kv_mb > 0 else 0.0

    result = {
        "model_name": model_name,
        "method": method,
        "sequence_length": seq_len,
        "kv_memory_mb": round(quant_kv_mb, 2),
        "baseline_kv_mb": round(fp16_kv_mb, 2),
        "compression_ratio": round(compression_ratio, 2)
    }

    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure KV Cache VRAM Delta & Compression Rate")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--seq_len", type=int, default=8192)
    args = parser.parse_args()

    metrics = measure_vram_usage(
        model_name=args.model_name,
        method=args.method,
        seq_len=args.seq_len
    )

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_vram_delta.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"VRAM delta and compression results successfully saved to {output_filename}")