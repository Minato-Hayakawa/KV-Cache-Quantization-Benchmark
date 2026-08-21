import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def get_optimal_device():
    """実行環境に合わせた最適なデバイスを自動判定"""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA / AMD GPU
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"   # Intel GPU
    else:
        return "cpu"   # CPU


def evaluate_fidelity(
    model,
    tokenizer,
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="rope_aware_tq",
    seq_len=1024,
    device="cuda",
):
    print(
        f"=== Evaluating Fidelity: {method} | Model: {model_id} | SeqLen: {seq_len} ==="
    )

    # 確実に seq_len に到達するよう十分な長さを確保
    base_text = "The rapid evolution of artificial intelligence and large language models has fundamentally transformed computation. "
    text = base_text * (seq_len // 10 + 10)
    inputs = tokenizer(
        text, return_tensors="pt", max_length=seq_len, truncation=True
    ).to(device)

    actual_seq_len = inputs.input_ids.size(1)
    print(f"Evaluated Sequence Length: {actual_seq_len} tokens")

    # RoPEを正しく適用するための position_ids 明示指定
    position_ids = torch.arange(0, actual_seq_len, dtype=torch.long, device=device).unsqueeze(0)

    # 1. FP16/BF16 ベースライン計算
    with torch.no_grad():
        baseline_cache = QuantizedKVCache(method="fp16")
        out_base = model(**inputs, position_ids=position_ids, past_key_values=baseline_cache, use_cache=True)
        logits_base = out_base.logits.float()  # 指標計算精度の安定化のため float32 化

    # 2. 量子化手法での計算
    with torch.no_grad():
        quant_cache = QuantizedKVCache(method=method)
        out_quant = model(**inputs, position_ids=position_ids, past_key_values=quant_cache, use_cache=True)
        logits_quant = out_quant.logits.float()  # 指標計算精度の安定化のため float32 化

    # --- 指標算出（単一トークンではなく全シーケンス位置での集約評価） ---
    
    # Cosine Similarity (全位置の平均)
    cos_sim = F.cosine_similarity(logits_base, logits_quant, dim=-1).mean().item()

    # Top-1 Match Rate (全位置の平均一致率)
    top1_base = logits_base.argmax(dim=-1)
    top1_quant = logits_quant.argmax(dim=-1)
    top1_match = (top1_base == top1_quant).float().mean().item() * 100

    # Top-5 Match Rate (ベクトル演算による全位置の平均 Top-5 重なり率)
    top5_base = logits_base.topk(5, dim=-1).indices
    top5_quant = logits_quant.topk(5, dim=-1).indices
    
    matches = (top5_base.unsqueeze(-1) == top5_quant.unsqueeze(-2)).any(dim=-1)
    top5_match = (matches.sum(dim=-1).float() / 5.0).mean().item() * 100

    print(f"Cosine Similarity : {cos_sim:.5f}")
    print(f"Top-1 Match Rate   : {top1_match:.2f}%")
    print(f"Top-5 Match Rate   : {top5_match:.2f}%\n")

    return {
        "cosine_similarity": cos_sim,
        "top1_match_rate": top1_match,
        "top5_match_rate": top5_match,
        "evaluated_seq_len": actual_seq_len
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fidelity for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Hugging Face model ID")
    parser.add_argument("--method", type=str, default="turbo_quant", help="Quantization method")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for evaluation")
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    # GPU実行時は Llama 3.1 等でのオーバーフローを防ぐため bfloat16 を使用
    torch_dtype = torch.float32 if target_device == "cpu" else torch.bfloat16

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch_dtype, device_map=target_device
        )
        .eval()
    )
    print("Model loaded successfully.\n")

    metrics = evaluate_fidelity(
        model,
        tokenizer,
        model_id=args.model_name,
        method=args.method,
        seq_len=args.seq_len,
        device=target_device,
    )
    
    print(f"Final Result [{args.model_name} | {args.method}]: {metrics}")

    # === PPL と合わせた出力ディレクトリ・命名規則での JSON 保存処理 ===
    results_data = {
        "model_name": args.model_name,
        "method": args.method,
        "seq_len": args.seq_len,
        **metrics
    }

    # 出力先ディレクトリの自動作成
    os.makedirs("results/ai-l40s", exist_ok=True)

    # モデル名に含まれるスラッシュをアンダースコアに置換（PPLスクリプトと完全同仕様）
    safe_model_name = args.model_name.replace("/", "_")
    
    # ファイル名がPPLと被らないよう、必要に応じてサフィックス（例: _fidelity）を付与
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_fidelity.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_filename}")