import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def get_optimal_device():
    """実行環境に合わせた最適なデバイスを自動判定"""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA および AMD (fs-mi300x)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"   # Intel GPU (qc-pvc)
    else:
        return "cpu"   # CPU


def run_niah(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="rope_aware_tq",
    context_len=8192,
    device=None,
):
    if device is None:
        device = get_optimal_device()

    print(
        f"=== Needle In A Haystack | Model: {model_id} | Context: {context_len} | Method: {method} | Device: {device} ==="
    )

    # 【重要】Llama 3.1 の長文オーバーフローを防ぐため GPU では bfloat16 を使用
    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
        )
        .eval()
    )

    needle = " The secret key for HPC cluster access is 998244353. "
    haystack_base = "The history of modern computing dates back to the mid-20th century. "

    tokens_per_phrase = len(tokenizer.encode(haystack_base, add_special_tokens=False))
    repeat_count = max(1, (context_len - 100) // tokens_per_phrase)

    # ランダム位置（Depth: 50% 付近）に Needle を挿入
    insert_pos = repeat_count // 2
    prompt = (
        (haystack_base * insert_pos)
        + needle
        + (haystack_base * (repeat_count - insert_pos))
    )
    prompt += "\nWhat is the secret key for HPC cluster access? Answer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    actual_prompt_len = inputs.input_ids.size(1)
    print(f"Actual Prompt Token Length: {actual_prompt_len}")

    cache = QuantizedKVCache(method=method)

    # --- Prefill ステップ（プロンプト処理と KV キャッシュ構築） ---
    with torch.no_grad():
        position_ids = torch.arange(0, actual_prompt_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model(
            input_ids=inputs.input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

    # 最初の生成トークン（Greedy: argmax）
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [next_token.item()]

    # --- Auto-regressive Decode ステップ（1トークンずつ生成） ---
    max_new_tokens = 15
    curr_pos = actual_prompt_len

    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        with torch.no_grad():
            pos_id = torch.tensor([[curr_pos]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=next_token,
                position_ids=pos_id,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            curr_pos += 1

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    success = "998244353" in response

    print(f"Output   : {response.strip()}")
    print(f"Success  : {success}\n")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Needle In A Haystack (NIAH) for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Hugging Face model ID")
    parser.add_argument("--method", type=str, default="rope_aware_tq", help="Quantization method")
    parser.add_argument("--context_len", type=int, default=8192, help="Context length for NIAH")
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    run_niah(
        model_id=args.model_name,
        method=args.method,
        context_len=args.context_len,
        device=target_device
    )