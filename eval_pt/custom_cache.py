import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from transformers.cache_utils import DynamicCache


class IdentityQuantizer:
    """
    passthrough 用の疑似量子化器。
    いかなる量子化も行わず、生テンソルをそのまま往復させる。
    キャッシュの配管（蓄積・連結・全履歴の返却）が正しいかを、
    量子化誤差の影響から分離して検証するためのもの（sanity_check.py で使用）。
    """

    designed_bits = 16.0

    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

    def decompress(self, compressed_data: torch.Tensor) -> torch.Tensor:
        return compressed_data


def _cat_compressed(existing, new):
    """
    量子化済みデータ同士を時系列方向 (dim=-2) に連結する。
    dict（手法ごとにスキーマが異なる）・裸テンソルの両方に対応し、
    キー名に依存せず「同一形状のテンソル同士」を連結する。
    """
    if isinstance(existing, torch.Tensor) and isinstance(new, torch.Tensor):
        return torch.cat([existing, new], dim=-2)

    if isinstance(existing, dict) and isinstance(new, dict):
        merged = {}
        keys = list(existing.keys()) + [k for k in new.keys() if k not in existing]
        for k in keys:
            e = existing.get(k)
            n = new.get(k)
            if e is None or n is None:
                merged[k] = n if e is None else e
            elif (
                isinstance(e, torch.Tensor)
                and isinstance(n, torch.Tensor)
                and e.ndim >= 2
                and e.shape[:-2] == n.shape[:-2]
                and e.shape[-1] == n.shape[-1]
                and e.dtype == n.dtype
            ):
                merged[k] = torch.cat([e, n], dim=-2)
            else:
                # スカラや dtype などのメタ情報は新しい側を採用
                merged[k] = n
        return merged

    return new


class QuantizedKVCache(DynamicCache):
    """
    Hugging Face transformers 評価用カスタム KV キャッシュクラス。

    設計方針:
      - 内部保持は「量子化済みデータ」(quantized_key_cache / quantized_value_cache)。
      - attention に渡す K/V は「全履歴デコード済みテンソル」(_deq_*_cache)。
        新規トークンぶんだけをデコードして時系列末尾に追記するため、
        デコード 1 ステップあたりの量子化オーバーヘッドは O(新規長) で済む。
      - 【重要】HF の各 attention 実装は past_key_values.update(...) の
        戻り値をそのまま K/V として使うため、戻り値は必ず【全履歴】でなければ
        ならない。デコード中に最新チャンクのみを返すと、モデルが
        「コンテキスト無しの1トークン attention」に陥る（旧バグ）。
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

        # 量子化済みの保存データ（手法ごとの dict またはテンソル）
        self.quantized_key_cache = []
        self.quantized_value_cache = []

        # attention 用の全履歴 K/V（高精度に戻したもの）
        self._deq_key_cache = []
        self._deq_value_cache = []

        # fp16 経路用（DynamicCache 本家のリスト）の存在担保
        if not hasattr(self, "key_cache"):
            self.key_cache = []
        if not hasattr(self, "value_cache"):
            self.value_cache = []

    def _lazy_init_quantizer(self, device, head_dim):
        if self.quantizer is not None or self.method == "fp16":
            return

        if self.method == "passthrough":
            self.quantizer = IdentityQuantizer()
        elif self.method == "turbo_quant":
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

        # fp16 / passthrough-OFF の場合は DynamicCache の標準処理
        if self.method == "fp16" or self.quantizer is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 1. 新規チャンクを量子化して保存
        q_key = self.quantizer.compress(key_states)
        q_value = self.quantizer.compress(value_states)

        # 2. 新規チャンクだけをデコードし、全履歴 K/V に追記
        d_key = self.quantizer.decompress(q_key)
        d_value = self.quantizer.decompress(q_value)

        while len(self.quantized_key_cache) <= layer_idx:
            self.quantized_key_cache.append(None)
            self.quantized_value_cache.append(None)
            self._deq_key_cache.append(None)
            self._deq_value_cache.append(None)

        if self.quantized_key_cache[layer_idx] is None:
            self.quantized_key_cache[layer_idx] = q_key
            self.quantized_value_cache[layer_idx] = q_value
            self._deq_key_cache[layer_idx] = d_key
            self._deq_value_cache[layer_idx] = d_value
        else:
            self.quantized_key_cache[layer_idx] = _cat_compressed(
                self.quantized_key_cache[layer_idx], q_key
            )
            self.quantized_value_cache[layer_idx] = _cat_compressed(
                self.quantized_value_cache[layer_idx], q_value
            )
            self._deq_key_cache[layer_idx] = torch.cat(
                [self._deq_key_cache[layer_idx], d_key], dim=-2
            )
            self._deq_value_cache[layer_idx] = torch.cat(
                [self._deq_value_cache[layer_idx], d_value], dim=-2
            )

        # 3. HF 側はこの戻り値をそのままアテンションの K/V として使う。
        #    必ず「全履歴」を返す（新旧バグ防止の核心）。
        return self._deq_key_cache[layer_idx], self._deq_value_cache[layer_idx]

    def __getitem__(self, layer_idx: int):
        # 量子化パスは自前の全履歴テンソルを返す
        if (
            self.method != "fp16"
            and self.quantizer is not None
            and len(self._deq_key_cache) > layer_idx
            and self._deq_key_cache[layer_idx] is not None
        ):
            return self._deq_key_cache[layer_idx], self._deq_value_cache[layer_idx]

        # fp16 / 未初期化時は DynamicCache 側のデータを返す。
        # transformers 4.x は __getitem__ あり、5.x 系は layers に格納。
        parent_getitem = getattr(super(), "__getitem__", None)
        if parent_getitem is not None:
            try:
                return parent_getitem(layer_idx)
            except (KeyError, IndexError):
                pass
        if hasattr(self, "layers") and len(self.layers) > layer_idx:
            layer = self.layers[layer_idx]
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                return layer.keys, layer.values
            if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
                return layer.key_cache, layer.value_cache
        raise KeyError(
            f"Cache has no data for layer {layer_idx}."
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.method == "fp16" or self.quantizer is None:
            return super().get_seq_length(layer_idx)
        if (
            len(self._deq_key_cache) <= layer_idx
            or self._deq_key_cache[layer_idx] is None
        ):
            return 0
        return self._deq_key_cache[layer_idx].shape[-2]

    def get_max_length(self, layer_idx: int = 0) -> int:
        """動的キャッシュなので上限なし（-1）を返す"""
        if self.method == "fp16" or self.quantizer is None:
            parent_fn = getattr(super(), "get_max_length", None)
            if parent_fn is not None:
                return parent_fn(layer_idx)
        return -1

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.get_max_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple:
        """
        transformers 5.x 系のマスク生成用。量子化パスでは親クラスの層管理を
        使っていないため、「過去長 + 今回のクエリ長」を自前リストから返す。
        """
        if self.method == "fp16" or self.quantizer is None:
            return super().get_mask_sizes(query_length, layer_idx)
        return query_length + self.get_seq_length(layer_idx), 0
