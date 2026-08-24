import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from transformers.cache_utils import DynamicCache


class QuantizedKVCache(DynamicCache):
    """
    Hugging Face transformers 評価用カスタム KV キャッシュクラス
    """

    def __init__(
        self,
        method: str = "turbo_quant",
        num_bits: int = 3,
        head_dim: int = None,
        bits_per_band: list = None,
    ):
        super().__init__()
        self.method = method
        self.num_bits = num_bits
        self.head_dim = head_dim
        self.bits_per_band = bits_per_band or [4, 2]
        self.quantizer = None  

        # 万が一親クラスに属性がない場合の保険として明示的に初期化
        if not hasattr(self, "key_cache"):
            self.key_cache = []
        if not hasattr(self, "value_cache"):
            self.value_cache = []

    def _lazy_init_quantizer(self, device, head_dim):
        if self.quantizer is not None or self.method == "fp16": 
            return
        
        if self.method == "turbo_quant":
            from core_pt.quantizers.turbo_quant import TurboQuantizer
            self.quantizer = TurboQuantizer(head_dim=head_dim, num_bits=self.num_bits, device=device.type)
        elif self.method == "rotor_quant":
            from core_pt.quantizers.rotor_quant import RotorQuantizer
            self.quantizer = RotorQuantizer(head_dim=head_dim, bits=self.num_bits, device=device.type)
        elif self.method == "hyper_quant":
            from core_pt.quantizers.hyper_quant import HyperQuantizer
            self.quantizer = HyperQuantizer(head_dim=head_dim, num_bits=self.num_bits, device=device.type)
        elif self.method == "ultra_quant":
            from core_pt.quantizers.ultra_quant import UltraQuantizer
            self.quantizer = UltraQuantizer(head_dim=head_dim, device=device.type)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        self._lazy_init_quantizer(
            device=key_states.device, head_dim=key_states.shape[-1]
        )

        # fp16 の場合、または量子化器がない場合は通常の DynamicCache の処理へ流す
        if self.method == "fp16" or self.quantizer is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 属性リストの存在を担保
        if not hasattr(self, "key_cache"):
            self.key_cache = []
        if not hasattr(self, "value_cache"):
            self.value_cache = []

        # 圧縮
        q_key = self.quantizer.compress(key_states)
        q_value = self.quantizer.compress(value_states)

        # DynamicCache の標準属性リスト（key_cache / value_cache）を利用
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(q_key)
            self.value_cache.append(q_value)
        else:
            existing_key = self.key_cache[layer_idx]
            existing_val = self.value_cache[layer_idx]
            
            if isinstance(existing_key, dict) and "quantized" in existing_key:
                cat_key = dict(existing_key)
                cat_val = dict(existing_val)
                for k in cat_key:
                    if isinstance(cat_key[k], torch.Tensor) and cat_key[k].ndim >= 2:
                        cat_key[k] = torch.cat([existing_key[k], q_key[k]], dim=-2)
                        cat_val[k] = torch.cat([existing_val[k], q_value[k]], dim=-2)
                self.key_cache[layer_idx] = cat_key
                self.value_cache[layer_idx] = cat_val
            else:
                self.key_cache.append(q_key)
                self.value_cache.append(q_value)

        return self

    def __getitem__(self, layer_idx: int):
        if self.method == "fp16" or self.quantizer is None or not hasattr(self, "key_cache") or len(self.key_cache) <= layer_idx:
            return super().__getitem__(layer_idx)

        q_key = self.key_cache[layer_idx]
        q_value = self.value_cache[layer_idx]

# 最後に key_states と value_states のタプルを返すように修正
        if isinstance(q_key, dict):
            return self.quantizer.decompress(q_key), self.quantizer.decompress(q_value)
        else:
            return q_key, q_value