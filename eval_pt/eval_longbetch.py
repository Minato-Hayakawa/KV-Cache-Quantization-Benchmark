import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache


def run_longbench_sample(
    model_id="meta-llama/Llama-3.2-1B",
    method="rope_aware_tq",
    device="cuda",
):
    print(f"=== LongBench Quick Eval | Method: {method} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device
        )
        .eval()
    )

    # 模擬タスク: 長文要約
    context = "Quantum computing relies on qubits instead of classical bits. Qubits leverage superposition and entanglement to execute complex calculations at speeds exponentially faster than conventional supercomputers. Key applications include cryptography, drug discovery, and financial modeling. " * 20
    instruction = "\nSummarize the core benefits of quantum computing in one concise sentence:"

    prompt = context + instruction
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=4096, truncation=True
    ).to(device)

    cache = QuantizedKVCache(method=method)

    with torch.no_grad():
        output = model.generate(
            **inputs, past_key_values=cache, max_new_tokens=40, use_cache=True
        )

    generated_text = tokenizer.decode(
        output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"Generated Summary: {generated_text.strip()}\n")


if __name__ == "__main__":
    for m in ["rope_aware_tq", "ultra_quant"]:
        run_longbench_sample(method=m)