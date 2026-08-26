import argparse
import json
import os
import sys
from transformers import AutoConfig

# core_pt を import するためのパス設定
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from eval_compression import designed_metadata_bytes  # noqa: E402

"""
解析的（GPU不要）の KV キャッシュ footprint 計算。

designed_bits は手書き定数ではなく、core_pt.quantizers の各実装を
CPU でインスタンス化して `designed_bits` 属性から導出する
（= 実装の量子化レベル数がそのまま論理ビット幅になる）。
"""


def get_designed_bits(method: str, head_dim: int, num_bits: int = 3) -> float:
    """実装している量子化器から論理ビット幅を導出する"""
    if method == "fp16":
        return 16.0
    if method == "turbo_quant":
        from core_pt.quantizers.turbo_quant import TurboQuantizer
        return TurboQuantizer(head_dim=head_dim, num_bits=num_bits, device="cpu").designed_bits
    if method == "rotor_quant":
        from core_pt.quantizers.rotor_quant import RotorQuantizer
        return RotorQuantizer(head_dim=head_dim, bits=num_bits, device="cpu").designed_bits
    if method == "hyper_quant":
        from core_pt.quantizers.hyper_quant import HyperQuantizer
        return HyperQuantizer(head_dim=head_dim, num_bits=num_bits, device="cpu").designed_bits
    if method == "ultra_quant":
        from core_pt.quantizers.ultra_quant import UltraQuantizer
        return UltraQuantizer(head_dim=head_dim, device="cpu").designed_bits
    raise ValueError(f"Unknown method: {method}")


def calculate_all_footprints(model_name, seq_len=8192, batch_size=1, num_bits=3):
    print(f"=== Calculating Analytical Footprints | Model: {model_name} | SeqLen: {seq_len} ===")

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    # head_dim は Mistral 系など「属性は存在するが None」の場合があるため、
    # getattr の既定値ではなく or でフォールバックする
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )

    total_elements = 2 * batch_size * num_layers * seq_len * num_kv_heads * head_dim
    fp16_mb = total_elements * 2 / (1024 * 1024)

    methods = ["fp16", "turbo_quant", "rotor_quant", "hyper_quant", "ultra_quant"]

    results = []
    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = model_name.replace("/", "_")

    for method in methods:
        bits = get_designed_bits(method, head_dim=head_dim, num_bits=num_bits)
        data_mb = total_elements * (bits / 8.0) / (1024 * 1024)
        meta_mb = designed_metadata_bytes(method, num_layers, num_kv_heads, head_dim, seq_len) / (1024 * 1024)
        footprint_mb = data_mb + meta_mb
        ratio = fp16_mb / footprint_mb if footprint_mb > 0 else 0.0

        metrics = {
            "model_name": model_name,
            "method": method,
            "designed_bits": bits,
            "sequence_length": seq_len,
            "designed_data_mb": round(data_mb, 2),
            "designed_meta_mb": round(meta_mb, 2),
            "designed_footprint_mb": round(footprint_mb, 2),
            "fp16_baseline_mb": round(fp16_mb, 2),
            "designed_compression_ratio": round(ratio, 2),
            "calculation_type": "analytical",
        }
        results.append(metrics)
        print(
            f"Method: {method:<12} | Bits: {bits:>4.1f} | Data: {data_mb:>8.2f} MB "
            f"| Meta: {meta_mb:>8.2f} MB | Total: {footprint_mb:>8.2f} MB | Compression: {ratio:>5.2f}x"
        )

        output_filename = f"results/ai-l40s/{safe_model_name}_{method}_theoretical.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    summary_filename = f"results/ai-l40s/{safe_model_name}_all_methods_footprint_summary.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nAll analytical footprint results successfully saved to results/ai-l40s/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analytical KV Cache Footprint (bits derived from implementations)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--seq_len", type=int, default=8192)
    args = parser.parse_args()

    calculate_all_footprints(
        model_name=args.model_name,
        seq_len=args.seq_len
    )
