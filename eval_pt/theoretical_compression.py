import argparse
import json
import os
from transformers import AutoConfig

def calculate_all_theoretical_compressions(model_name, seq_len=8192, batch_size=1):
    print(f"=== Calculating Theoretical Compression for All Methods | Model: {model_name} | SeqLen: {seq_len} ===")
    
    # モデルのコンフィグから構造を取得（GPU不要）
    config = AutoConfig.from_pretrained(model_name)
    
    num_hidden_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", 32))
    num_key_value_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32))
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads

    # KVキャッシュの総要素数（KeyとValueの2つ分）
    total_elements = 2 * batch_size * num_hidden_layers * seq_len * num_key_value_heads * head_dim

    # FP16（非圧縮）の理論サイズ（バイト）
    fp16_bits = 16.0
    baseline_bytes = total_elements * (fp16_bits / 8.0)
    baseline_mb = baseline_bytes / (1024 * 1024)

    # 各アルゴリズムの設計ビット数（論文や仕様に基づく標準的なビットレート）
    # ※ 必要に応じて各手法のビット数を調整してください
    method_bits = {
        "fp16": 16.0,
        "turbo_quant": 3.0,   # 例: TurboQuantは通常3-bit等
        "rotor_quant": 3.0,   # 例: RotorQuant
        "hyper_quant": 2.5,   # 例: HyperQuant
        "ultra_quant": 2.0    # 例: UltraQuant
    }

    results = []
    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = model_name.replace("/", "_")

    for method, bits in method_bits.items():
        quant_bytes = total_elements * (bits / 8.0)
        quant_mb = quant_bytes / (1024 * 1024)
        compression_ratio = baseline_mb / quant_mb if quant_mb > 0 else 0.0

        metrics = {
            "model_name": model_name,
            "method": method,
            "designed_bits": bits,
            "sequence_length": seq_len,
            "kv_cache_size_mb": round(quant_mb, 2),
            "baseline_mb": round(baseline_mb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "calculation_type": "theoretical"
        }
        results.append(metrics)
        print(f"Method: {method:<12} | Bits: {bits} | Size: {quant_mb:>8.2f} MB | Compression: {compression_ratio:>5.2f}x")

        # 個別JSONとしても保存
        output_filename = f"results/ai-l40s/{safe_model_name}_{method}_theoretical.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    # まとめて一覧用のJSONとしても保存
    summary_filename = f"results/ai-l40s/{safe_model_name}_all_methods_theoretical_summary.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nAll theoretical compression results successfully saved to results/ai-l40s/")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Theoretical KV Cache Compression Rate for All Methods")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--seq_len", type=int, default=8192)
    args = parser.parse_args()

    calculate_all_theoretical_compressions(
        model_name=args.model_name,
        seq_len=args.seq_len
    )