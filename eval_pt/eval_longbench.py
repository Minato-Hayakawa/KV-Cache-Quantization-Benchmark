import argparse
import json
import os
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
        return "cpu"   # 富岳等 (fx700)


def run_longbench_sample(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    method="rope_aware_tq",
    device=None,
):
    if device is None:
        device = get_optimal_device()

    print(f"=== LongBench Quick Eval | Model: {model_id} | Method: {method} | Device: {device} ===")

    # CPU環境（富岳など）の場合は、安定性を考慮して float32 を使用
    torch_dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
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
    ).strip()
    
    print(f"Generated Summary: {generated_text}\n")

    return {
        "generated_summary": generated_text,
        "input_length": inputs.input_ids.shape[1]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LongBench Quick Sample for KV Cache Quantization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Hugging Face model ID")
    parser.add_argument("--method", type=str, default="rope_aware_tq", help="Quantization method")
    args = parser.parse_args()

    target_device = get_optimal_device()
    print(f"Detected Active Device: {target_device}\n")

    metrics = run_longbench_sample(
        model_id=args.model_name,
        method=args.method,
        device=target_device
    )
    
    print(f"Final Result [{args.model_name} | {args.method}]: {metrics}")

    # === 他の評価スクリプトと統一された出力先・命名規則による JSON 保存処理 ===
    results_data = {
        "model_name": args.model_name,
        "method": args.method,
        **metrics
    }

    # 出力先ディレクトリの自動作成
    os.makedirs("results/ai-l40s", exist_ok=True)

    # モデル名に含まれるスラッシュをアンダースコアに置換
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_{args.method}_longbench.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_filename}")