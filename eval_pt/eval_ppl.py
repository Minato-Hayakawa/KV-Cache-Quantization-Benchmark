import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def get_optimal_device():
    """実行環境に合わせた最適なデバイスを自動判定"""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA (b300, a100, h200等) および AMD (fs-mi300x)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"   # Intel GPU (qc-pvc)
    else:
        return "cpu"   # 富岳等 (fx700)


def evaluate_ppl(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="rope_aware_tq",
    stride=512,
    device=None,
):
    if device is None:
        device = get_optimal_device()

    print(f"=== Evaluating PPL: {method} | Device: {device} | Model: {model_id} ===")

    # CPU環境（富岳など）の場合は float32、GPU環境は float16
    torch_dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
        )
        .eval()
    )

    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    max_length = 2048
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        cache = QuantizedKVCache(method=method)

        with torch.no_grad():
            outputs = model(input_ids, past_key_values=cache, use_cache=True)
            
            # 【修正点】Logit[t] と Target[t+1] を比較するための Shift 処理
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            neg_log_likelihood = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,  # マスク(-100)のトークンは損失計算から除外
            )

        num_valid_tokens = (shift_labels != -100).sum().item()
        nll_sum += neg_log_likelihood.item()
        n_tokens += num_valid_tokens
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    
    print(f"[{method}] Perplexity: {ppl:.4f}\n")
    return ppl


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Evaluate PPL for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Hugging Face model ID")
    parser.add_argument("--method", type=str, default="turbo_quant", help="Quantization method")
    parser.add_argument("--stride", type=int, default=512, help="Stride for evaluation")
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    # 引数で指定されたモデルと手法で実行
    ppl_result = evaluate_ppl(
        model_id=args.model_name,
        method=args.method,
        stride=args.stride,
        device=target_device
    )
    print(f"Final Result [{args.model_name} | {args.method}]: {ppl_result}")