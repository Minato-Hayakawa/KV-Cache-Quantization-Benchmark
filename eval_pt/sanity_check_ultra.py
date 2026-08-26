import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_cache import QuantizedKVCache
from sanity_check import greedy_generate, get_optimal_device
from core_pt.quantizers.ultra_quant import UltraQuantizer

"""
ultra_quant 専用の sanity 実験（本家 eval_pt/sanity_check.py の補完）。

背景:
  ultra_quant は FP4 (E2M1) 16点固定グリッドが手法の定義そのものであり、
  turbo/rotor/hyper のような「同じ経路のまま 7bit に上げる」高ビット
  モードが存在しない。したがって本家の 7bit 往復テストには参加できない。

本スクリプトでは代わりに以下の2検証を行う:

  A. ロスレス分解テスト (HP bypass 生成テスト)
     FP4グリッドへの丸めのみを外した高精密バイパス版
     (WHT回転と absmax/6 スケール往復は本番と同一、格納は実質 fp32)
     で greedy 16トークン生成し、fp16 とのトークン一致率 >= 0.9 を要求。
     回転は直交のため往復誤差は fp32 精度でほぼゼロであり、一致率は
     1.0 前後が期待値。これを大きく外す場合は往復の実装破綻を疑う。
     → 回転・スケール・キャッシュ配管の数式的健全性の検証。

  B. ネイティブ4bit グリッド健全性チェック (モデル不要の数値検証)
     seed 固定の乱数テンソルを本番そのままの4bitモードで
     compress→decompress し、
       (a) 相対L2誤差 < 0.40
          （本番KVの実測 0.18〜0.30 にマージン。index負数wrap等の
           破綻時は 0.7〜1.4 まで跳ね上がるため明確に検出可能）
       (b) 全量子化インデックスが [0, 15] 内
          （int8 溢れ型の静かな破損を直接検出）
     → FP4グリッドマッピング自体の検証。ロスレス性では判定できないため
       誤差上限の定数として検証する。

干渉防止の設計:
  既存コードの変更はゼロ。bypass 量子化器・キャッシュ差し替えともに
  本スクリプト内の継承クラスで実現する。結果の出力先も本家 sanity
  ({model}_sanity.json) とは別ファイル ({model}_sanity_ultra.json)。
"""

HP_MATCH_THRESHOLD = 0.9
NATIVE_REL_L2_THRESHOLD = 0.40
NATIVE_INDEX_MIN, NATIVE_INDEX_MAX = 0, 15  # FP4 E2M1 グリッドは16点

# 本家 sanity_check.py と同一の固定プロンプト（比較可能性のため）
PROMPT = (
    "The history of modern computing dates back to the mid-20th century. "
    "Quantum computing leverages superposition and entanglement. "
) * 12
PROMPT += "Question: What does quantum computing leverage? Answer:"


class UltraBypassQuantizer(UltraQuantizer):
    """
    sanity 専用: FP4グリッドへの丸めのみを外したロスレス分解版。

    WHT 回転・absmax/6 スケール往復は本番 UltraQuantizer と同一。
    回転後のスケール済み実数を fp32 のまま保持する（量子化丸めなし）。
    グリッドの argmin / index 格納を経由しないため、このパスで
    壊れればバイパス以外（回転・スケール・配管）の破綻と特定できる。
    """

    def compress(self, tensor: torch.Tensor) -> dict:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)

        H = self.hadamard_matrix.to(tensor.device)
        transformed = torch.matmul(tensor_f32, H)

        max_val = transformed.abs().max(dim=-1, keepdim=True).values
        scale = max_val / 6.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        scaled_val = transformed / scale

        return {
            "rot_scaled": scaled_val,
            "scale": scale,
            "dtype": orig_dtype,
        }

    def decompress(self, compressed_data: dict) -> torch.Tensor:
        scaled_val = compressed_data["rot_scaled"]
        scale = compressed_data["scale"]
        orig_dtype = compressed_data["dtype"]

        transformed = scaled_val * scale
        H = self.hadamard_matrix.to(transformed.device)
        recovered = torch.matmul(transformed, H.T)
        return recovered.to(orig_dtype)


class UltraSanityKVCache(QuantizedKVCache):
    """bypass 版量子化器を差し込むだけのサブクラス（本番経路は変更しない）。"""

    def _lazy_init_quantizer(self, device, head_dim):
        if self.method == "ultra_quant_bypass":
            self.quantizer = UltraBypassQuantizer(
                head_dim=head_dim, device=device.type
            )
            return
        super()._lazy_init_quantizer(device, head_dim)


def check_native_4bit_roundtrip(device, n_rows=64, head_dim=128, seed=0):
    """検証B: 本番4bitモードの数値健全性（モデル不要）"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_rows, head_dim, generator=gen).to(device)

    quantizer = UltraQuantizer(head_dim=head_dim, device=device)
    data = quantizer.compress(x)

    idx = data["quantized_idx"]
    idx_min = int(idx.min().item())
    idx_max = int(idx.max().item())
    index_in_range = (idx_min >= NATIVE_INDEX_MIN) and (idx_max <= NATIVE_INDEX_MAX)

    rec = quantizer.decompress(data)
    rel_l2 = float((rec - x).norm().item() / x.norm().item())

    passed = index_in_range and rel_l2 < NATIVE_REL_L2_THRESHOLD
    return {
        "rel_l2": rel_l2,
        "index_min": idx_min,
        "index_max": idx_max,
        "index_in_range": index_in_range,
        "criterion": (
            f"rel_l2 < {NATIVE_REL_L2_THRESHOLD} and "
            f"all indices in [{NATIVE_INDEX_MIN}, {NATIVE_INDEX_MAX}]"
        ),
        "pass": passed,
    }


def run_ultra_sanity(model_id, max_new_tokens=16, device=None):
    if device is None:
        device = get_optimal_device()

    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"=== ultra_quant Dedicated Sanity | Model: {model_id} | Device: {device} ===")

    # --- 検証B: ネイティブ4bit（モデル不要、先に実行） ---
    native = check_native_4bit_roundtrip(device)
    print(
        f"[native 4bit] rel L2 = {native['rel_l2']:.4f} "
        f"(< {NATIVE_REL_L2_THRESHOLD}), "
        f"index range = [{native['index_min']}, {native['index_max']}] "
        f"-> {'PASS' if native['pass'] else 'FAIL'}"
    )

    # --- 検証A: HP bypass 生成テスト ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
        )
        .eval()
    )

    fp16_cache = UltraSanityKVCache(method="fp16")
    fp16_ids = greedy_generate(model, tokenizer, PROMPT, fp16_cache, max_new_tokens, device)
    print(f"[fp16]            generated: {fp16_ids}")

    hp_cache = UltraSanityKVCache(method="ultra_quant_bypass")
    hp_ids = greedy_generate(model, tokenizer, PROMPT, hp_cache, max_new_tokens, device)
    n = min(len(fp16_ids), len(hp_ids))
    match_rate = sum(a == b for a, b in zip(fp16_ids[:n], hp_ids[:n])) / max(1, n)
    hp_pass = match_rate >= HP_MATCH_THRESHOLD
    print(f"[ultra HP bypass] generated: {hp_ids}")
    print(
        f"[ultra HP bypass] match rate vs fp16: {match_rate*100:.1f}% "
        f"(>= {HP_MATCH_THRESHOLD*100:.0f}%) -> {'PASS' if hp_pass else 'FAIL'}"
    )

    overall_pass = bool(native["pass"] and hp_pass)
    print(f"\n=== ultra_quant Sanity Overall: {'PASS' if overall_pass else 'FAIL'} ===")

    return {
        "model_name": model_id,
        "overall_pass": overall_pass,
        "checks": {
            "ultra_hp_bypass_generation": {
                "generated_ids_fp16": fp16_ids,
                "generated_ids_hp_bypass": hp_ids,
                "token_match_rate_vs_fp16": match_rate,
                "criterion": (
                    f"token_match_rate >= {HP_MATCH_THRESHOLD} "
                    "(lossless rotation/scale round-trip; FP4 grid snap bypassed)"
                ),
                "pass": hp_pass,
            },
            "ultra_native_4bit_roundtrip": native,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ultra_quant 専用 sanity（HP bypass 生成テスト + ネイティブ4bit 数値検証）"
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    metrics = run_ultra_sanity(model_id=args.model_name, max_new_tokens=args.max_new_tokens)

    os.makedirs("results/ai-l40s", exist_ok=True)
    safe_model_name = args.model_name.replace("/", "_")
    output_filename = f"results/ai-l40s/{safe_model_name}_sanity_ultra.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Results successfully saved to {output_filename}")

    # 次工程の評価を無効な状態で進めないよう、失敗時は非ゼロ終了
    if not metrics["overall_pass"]:
        raise SystemExit(1)
