import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def evaluate_fidelity(
    model,
    tokenizer,
    model_id="meta-llama/Llama-3.2-1B",
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
    model_id = "meta-llama/Llama-3.2-1B"
    device = "cuda"
    
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device
        )
        .eval()
    )
    print("Model loaded successfully.\n")

    for m in ["turbo_quant", "rope_aware_tq", "hyper_quant", "ultra_quant"]:
        evaluate_fidelity(model, tokenizer, model_id=model_id, method=m, device=device)