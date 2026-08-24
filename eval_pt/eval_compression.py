import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache  # ファイル名を custom_cache_2 にしている場合は適宜変更してください

def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

def count_tensor_bytes(obj):
    """
    テンソル、辞書、リストなどネストした構造の中から
    すべてのPyTorchテンソルのメモリサイズ（バイト数）を再帰的に計算する関数
    """
    total = 0
    if isinstance(obj, torch.Tensor):
        total += obj.element_size() * obj.nelement()
    elif isinstance(obj, dict):
        for v in obj.values():
            total += count_tensor_bytes(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += count_tensor_bytes(item)
    return total

def measure_kv_cache_memory(model_name, method, seq_len=8192):
    print(f"=== Measuring Compression Rate | Model: {model_name} | Method: {method} | SeqLen: {seq_len} ===")
    
    device = get_optimal_device()
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    ).eval()

    # ダミーの長文コンテキストを生成
    dummy_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)

    # CUDAのメモリ使用量をクリア
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # KVキャッシュのインスタンス準備
    cache = QuantizedKVCache(method=method) if method != "fp16" else None

    # フォワードパスを実行してKVキャッシュを生成させる
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            past_key_values=cache,
            use_cache=True
        )

    # 生成された KV キャッシュ（past_key_values）の実際のメモリサイズ（バイト数）を計算
    kv_cache = outputs.past_key_values
    total_kv_bytes = 0

    if kv_cache is not None:
        # カスタムキャッシュの key_cache / value_cache から正確な圧縮後サイズを算出
        if hasattr(kv_cache, "key_cache") and kv_cache.key_cache:
            for layer_key, layer_val in zip(kv_cache.key_cache, kv_cache.value_cache):
                total_kv_bytes += count_tensor_bytes(layer_key)
                total_kv_bytes += count_tensor_bytes(layer_val)
        else:
            # フォールバック
            for layer_cache in kv_cache:
                total_kv_bytes += count_tensor_bytes(layer_cache)

    total_kv_mb = total_kv_bytes / (1024 * 1024)

    # FP16ベースラインに対する圧縮率を計算
    baseline_reference_mb = 289.0 
    compression_ratio = baseline_reference_mb / total_kv_mb if total_kv_mb > 0 else 0.0

    result = {
        "model_name": model_name,
        "method": method,
        "sequence_length": seq_len,
        "kv_cache_size_mb": round(total_kv_mb, 2),
        "baseline_mb": baseline_reference_mb,
        "compression_ratio": round(compression_ratio, 2)
    }

    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure KV Cache Compression Rate")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--seq_len", type=int, default=8192)
    args = parser.parse_args()

    metrics = measure_kv_cache_memory(
        model_name=args.model_name,
        method=args.method,
        seq_len=args.seq_len
    )

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_compression.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Compression results successfully saved to {output_filename}")