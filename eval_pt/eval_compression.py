import argparse
import json
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache

"""
KV キャッシュのメモリ評価（2系統の数字を明確に分離して出力する）:

1. stored_bytes（この簡易実装の実際の保存占有）
   - シミュレーション用に int8 量子化値 + fp32 スケールで保持したテンソルの
     実バイト数を、forward 後のキャッシュオブジェクトから再帰的に数える。
   - ※ これは「このPython簡易実装の保存占有」であって、designed_bits を
     反映しない（全手法 int8 で保存される）ため、手法間の設計ビット差を
     議論するために使ってはいけない。

2. designed footprint（解析的 footprint）
   - 各量子化器の designed_bits（実装のレベル数から導出）から
     データ部 (= 全要素数 × bits / 8) を解析的に計算し、
     実行時メタデータ（スケール等）を実装仕様からモデル化して加算したもの。
   - 本番環境の実測メモリではなく「この実装の設計どおりにパッキングした
     場合の理論値」として扱う。
"""


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def count_tensor_bytes(obj):
    """テンソル・dict・list などネスト構造内の全テンソルのバイト数を再帰集計"""
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


def kv_geometry(model_name):
    """config から KV キャッシュの幾何情報を取得"""
    config = AutoConfig.from_pretrained(model_name)
    num_layers = getattr(config, "num_hidden_layers")
    num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    # head_dim は Mistral 系など「属性は存在するが None」の場合があるため、
    # getattr の既定値ではなく or でフォールバックする
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return num_layers, num_kv_heads, head_dim


def designed_metadata_bytes(method, num_layers, num_kv_heads, head_dim, seq_len):
    """実装仕様に基づくランタイムメタデータ（スケール等）の解析的バイト数"""
    kb = 2 * num_layers * num_kv_heads * seq_len  # (K,V) × 層 × ヘッド × トークン
    if method in ("turbo_quant", "hyper_quant", "ultra_quant", "passthrough"):
        # per-token ベクトルあたり fp32 スケール 1 個
        return kb * 4
    if method == "rotor_quant":
        num_blocks = head_dim // 3
        tail = head_dim - num_blocks * 3
        # 3D ブロックごとに fp32 スケール + 非回転の尻尾は fp32 生保持
        return kb * (num_blocks * 4 + tail * 4)
    if method == "fp16":
        return 0
    raise ValueError(f"Unknown method: {method}")


def measure_compression(model_name, method, seq_len=8192):
    print(f"=== Measuring KV Cache Compression | Model: {model_name} | Method: {method} | SeqLen: {seq_len} ===")

    device = get_optimal_device()
    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    ).eval()

    dummy_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
    actual_seq_len = inputs.input_ids.size(1)

    cache = QuantizedKVCache(method=method)
    with torch.no_grad():
        _ = model(
            input_ids=inputs.input_ids,
            past_key_values=cache,
            use_cache=True,
        )

    # --- 1. 保存占有 (stored bytes) ---
    if method == "fp16":
        num_layers, _, _ = kv_geometry(model_name)
        stored_bytes = 0
        for li in range(num_layers):
            k, v = cache[li]
            stored_bytes += count_tensor_bytes(k) + count_tensor_bytes(v)
        designed_bits = 16.0
    else:
        stored_bytes = sum(
            count_tensor_bytes(qk) + count_tensor_bytes(qv)
            for qk, qv in zip(cache.quantized_key_cache, cache.quantized_value_cache)
        )
        designed_bits = cache.quantizer.designed_bits

    stored_mb = stored_bytes / (1024 * 1024)

    # --- 2. 解析的 designed footprint ---
    num_layers, num_kv_heads, head_dim = kv_geometry(model_name)
    total_elements = 2 * num_layers * num_kv_heads * actual_seq_len * head_dim
    designed_data_mb = total_elements * (designed_bits / 8.0) / (1024 * 1024)
    meta_bytes = designed_metadata_bytes(method, num_layers, num_kv_heads, head_dim, actual_seq_len)
    designed_meta_mb = meta_bytes / (1024 * 1024)
    designed_footprint_mb = designed_data_mb + designed_meta_mb

    result = {
        "model_name": model_name,
        "method": method,
        "sequence_length": actual_seq_len,
        "designed_bits": designed_bits,
        "stored_mb": round(stored_mb, 2),
        "designed_data_mb": round(designed_data_mb, 2),
        "designed_meta_mb": round(designed_meta_mb, 2),
        "designed_footprint_mb": round(designed_footprint_mb, 2),
        "note": (
            "stored_mb は簡易実装(int8保存)の実占有で設計ビットを反映しない。"
            "designed_footprint_mb は(designed_bits基準データ+実装メタデータ)の解析値。"
        ),
    }
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure KV Cache stored bytes & designed footprint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--seq_len", type=int, default=8192)
    args = parser.parse_args()

    metrics = measure_compression(
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
