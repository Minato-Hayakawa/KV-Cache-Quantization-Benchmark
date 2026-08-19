import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def run_niah(
    model_id="meta-llama/Llama-3.2-1B",
    method="rope_aware_tq",
    context_len=8192,
    device="cuda",
):
    print(
        f"=== Needle In A Haystack | Context: {context_len} | Method: {method} ==="
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device
        )
        .eval()
    )

    needle = " The secret key for HPC cluster access is 998244353. "
    haystack_base = "The history of modern computing dates back to the mid-20th century. "

    tokens_per_phrase = len(tokenizer.encode(haystack_base))
    repeat_count = (context_len - 100) // tokens_per_phrase

    # ランダム位置（Depth: 50% 付近）に Needle を挿入
    insert_pos = repeat_count // 2
    prompt = (
        (haystack_base * insert_pos)
        + needle
        + (haystack_base * (repeat_count - insert_pos))
    )
    prompt += "\nWhat is the secret key for HPC cluster access? Answer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    cache = QuantizedKVCache(method=method)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, past_key_values=cache, max_new_tokens=15, use_cache=True
        )

    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    success = "998244353" in response

    print(f"Output   : {response.strip()}")
    print(f"Success  : {success}\n")
    return success


if __name__ == "__main__":
    for m in ["fp16", "rope_aware_tq", "ultra_quant"]:
        run_niah(method=m)