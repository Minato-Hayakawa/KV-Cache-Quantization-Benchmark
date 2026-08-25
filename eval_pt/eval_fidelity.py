import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache

"""
忠実度評価（Simulated Quantization 下での品質指標）。

旧版の問題点:
  - 最終 logits を比較していたのに「アテンション出力の忠実度」と
    README に記載されていた。
  - Top-5 は「ベースライン top-5 のうち量子化側 top-5 に含まれる割合」
    （重なり率）で、名称から直感的な意味と異なっていた。

本バージョンの指標:
  - logit_cosine / logit_top1 / logit_top5_overlap ... 最終 logits 空間の指標
  - kv_cosine / kv_relative_mse ... キャッシュの K/V ベクトルそのものとの比較
    （量子化誤差がモデルの非線形で増幅される前の純粋な尺度）
"""


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def _num_layers(model):
    return model.config.num_hidden_layers


def evaluate_fidelity(
    model,
    tokenizer,
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="turbo_quant",
    seq_len=1024,
    device="cuda",
):
    print(
        f"=== Evaluating Fidelity: {method} | Model: {model_id} | SeqLen: {seq_len} ==="
    )

    base_text = "The rapid evolution of artificial intelligence and large language models has fundamentally transformed computation. "
    text = base_text * (seq_len // 10 + 10)
    inputs = tokenizer(
        text, return_tensors="pt", max_length=seq_len, truncation=True
    ).to(device)

    actual_seq_len = inputs.input_ids.size(1)
    print(f"Evaluated Sequence Length: {actual_seq_len} tokens")

    position_ids = torch.arange(0, actual_seq_len, dtype=torch.long, device=device).unsqueeze(0)

    # 1. FP16/BF16 ベースライン
    with torch.no_grad():
        baseline_cache = QuantizedKVCache(method="fp16")
        out_base = model(**inputs, position_ids=position_ids, past_key_values=baseline_cache, use_cache=True)
        logits_base = out_base.logits.float()

    # 2. 量子化手法
    with torch.no_grad():
        quant_cache = QuantizedKVCache(method=method)
        out_quant = model(**inputs, position_ids=position_ids, past_key_values=quant_cache, use_cache=True)
        logits_quant = out_quant.logits.float()

    # --- 最終 logits 空間の指標 ---
    logit_cosine = F.cosine_similarity(logits_base, logits_quant, dim=-1).mean().item()

    top1_base = logits_base.argmax(dim=-1)
    top1_quant = logits_quant.argmax(dim=-1)
    logit_top1 = (top1_base == top1_quant).float().mean().item() * 100

    top5_base = logits_base.topk(5, dim=-1).indices
    top5_quant = logits_quant.topk(5, dim=-1).indices
    matches = (top5_base.unsqueeze(-1) == top5_quant.unsqueeze(-2)).any(dim=-1)
    logit_top5_overlap = (matches.sum(dim=-1).float() / 5.0).mean().item() * 100

    # --- KV ベクトル空間の指標（quantize→dequantize の直接誤差） ---
    kv_cos_list = []
    kv_mse_list = []
    for li in range(_num_layers(model)):
        k_base, v_base = baseline_cache[li]
        k_quant, v_quant = quant_cache[li]
        for kb, kq in ((k_base, k_quant), (v_base, v_quant)):
            kbf = kb.reshape(-1, kb.shape[-1]).float()
            kqf = kq.reshape(-1, kq.shape[-1]).float()
            kv_cos_list.append(F.cosine_similarity(kbf, kqf, dim=-1).mean().item())
            rel = ((kbf - kqf).norm() / (kbf.norm() + 1e-12)).item()
            kv_mse_list.append(rel)

    kv_cosine = sum(kv_cos_list) / len(kv_cos_list)
    kv_relative_mse = sum(kv_mse_list) / len(kv_mse_list)

    print(f"Logit cos-sim        : {logit_cosine:.5f}")
    print(f"Logit Top-1 match    : {logit_top1:.2f}%")
    print(f"Logit Top-5 overlap  : {logit_top5_overlap:.2f}%")
    print(f"KV cos-sim           : {kv_cosine:.5f}")
    print(f"KV relative L2 error : {kv_relative_mse:.5f}\n")

    return {
        "logit_cosine_similarity": logit_cosine,
        "logit_top1_match_rate": logit_top1,
        "logit_top5_overlap_rate": logit_top5_overlap,
        "kv_cosine_similarity": kv_cosine,
        "kv_relative_mse": kv_relative_mse,
        "evaluated_seq_len": actual_seq_len,
        "note": "logit_* は最終 logits 空間、kv_* はキャッシュ K/V ベクトル空間の忠実度",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fidelity (logit + KV) for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--method", type=str, default="turbo_quant")
    parser.add_argument("--seq_len", type=int, default=1024)
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

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

    results_data = {
        "model_name": args.model_name,
        "method": args.method,
        "seq_len": args.seq_len,
        **metrics
    }

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_fidelity.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_filename}")
