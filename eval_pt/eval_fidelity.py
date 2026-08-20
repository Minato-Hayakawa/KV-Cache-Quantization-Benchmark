import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def get_optimal_device():
    """実行環境に合わせた最適なデバイスを自動判定"""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA および AMD (fs-mi300x)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"   # Intel GPU (qc-pvc)
    else:
        return "cpu"   # 富岳等 (fx700)


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

    text = "The rapid evolution of artificial intelligence and large language models has fundamentally transformed computation." * 30
    inputs = tokenizer(
        text, return_tensors="pt", max_length=seq_len, truncation=True
    ).to(device)

    # 1. FP16 ベースライン
    with torch.no_grad():
        baseline_cache = QuantizedKVCache(method="fp16")
        out_base = model(**inputs, past_key_values=baseline_cache, use_cache=True)
        logits_base = out_base.logits[:, -1, :]

    # 2. 量子化手法
    with torch.no_grad():
        quant_cache = QuantizedKVCache(method=method)
        out_quant = model(**inputs, past_key_values=quant_cache, use_cache=True)
        logits_quant = out_quant.logits[:, -1, :]

    # 指標算出
    cos_sim = F.cosine_similarity(
        logits_base.float(), logits_quant.float(), dim=-1
    ).mean().item()

    top1_base = logits_base.argmax(dim=-1)
    top1_quant = logits_quant.argmax(dim=-1)
    top1_match = (top1_base == top1_quant).float().mean().item() * 100

    top5_base = logits_base.topk(5, dim=-1).indices
    top5_quant = logits_quant.topk(5, dim=-1).indices
    
    top5_base_set = set(top5_base[0].tolist())
    top5_quant_set = set(top5_quant[0].tolist())
    top5_match = (len(top5_quant_set & top5_base_set) / 5.0) * 100

    print(f"Cosine Similarity : {cos_sim:.5f}")
    print(f"Top-1 Match Rate  : {top1_match:.2f}%")
    print(f"Top-5 Match Rate  : {top5_match:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fidelity for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Hugging Face model ID")
    parser.add_argument("--method", type=str, default="turbo_quant", help="Quantization method")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for evaluation")
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    # CPU環境の場合は安定性を考慮して float32 を使用
    torch_dtype = torch.float32 if target_device == "cpu" else torch.float16

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch_dtype, device_map=target_device
        )
        .eval()
    )
    print("Model loaded successfully.\n")

    evaluate_fidelity(
        model,
        tokenizer,
        model_id=args.model_name,
        method=args.method,
        seq_len=args.seq_len,
        device=target_device,
    )