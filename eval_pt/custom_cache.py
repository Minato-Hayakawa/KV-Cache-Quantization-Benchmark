import sys
import os

# 1. プロジェクトのルートディレクトリへのパスを通す
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from transformers.cache_utils import DynamicCache


class QuantizedKVCache(DynamicCache):
    """
    Hugging Face transformers 評価用カスタム KV キャッシュクラス
    指定したアルゴリズムで Key / Value テンソルを圧縮・保持し、
    モデルから参照される際に動的に復元（デコード）します。
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

    def _lazy_init_quantizer(self, device, head_dim):
        if self.quantizer is not None: 
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
        """
        key_states, value_states shape: (batch_size, num_heads, seq_len, head_dim)
        """
        self._lazy_init_quantizer(
            device=key_states.device, head_dim=key_states.shape[-1]
        )

        # FP16 (ベースライン) の場合は通常の親クラス処理へ
        if self.quantizer is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 1. 復元せず「圧縮（compress）」のみを実行して辞書データを得る
        q_key = self.quantizer.compress(key_states)
        q_value = self.quantizer.compress(value_states)

        # 2. 親クラスのキャッシュリスト（key_cache / value_cache）に圧縮データを格納
        # （※DynamicCacheの仕様に合わせてインデックスを管理）
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(q_key)
            self.value_cache.append(q_value)
        else:
            # シーケンスが長くなった場合は沿うように結合（各量子化器の形式に依存するためテンソル部分を結合）
            # ここではシンプルにリストへ追加・拡張する標準的なハンドリングを行います
            existing_key = self.key_cache[layer_idx]
            existing_val = self.value_cache[layer_idx]
            
            # 辞書の中の主要テンソルを seq_len 方向（通常 dim=-2）で結合
            if isinstance(existing_key, dict) and "quantized" in existing_key:
                # 共通の結合処理
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
        """
        モデルがキャッシュを参照してアテンション計算を行う際、
        自動的に保持している圧縮データをデコード（復元）して返す。
        """
        if self.quantizer is None or len(self.key_cache) <= layer_idx:
            return super().__getitem__(layer_idx)

        q_key = self.key_cache[layer_idx]
        q_value = self.value_cache[layer_idx]

        # 辞書データであればデコード、すでにテンソルならそのまま返す
        if isinstance(q_key, dict):
            key_states = self.quantizer.decompress(q_key)
            value_states = self.quantizer.decompress(q_value)
            return key_states, value_states
        else:
            return q_key, q_value