import sys
import os

# 1. プロジェクトのルートディレクトリ（KV-Cache Quantization Benchmark）へのパスを通す
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from transformers.cache_utils import DynamicCache

# 2. ルートからの絶対インポート
from core_pt.quantizers.base import BaseQuantizer
from core_pt.quantizers.turbo_quant import TurboQuantizer
from core_pt.quantizers.rope_aware_tq import RoPEAwareTurboQuantizer
from core_pt.quantizers.hyper_quant import HyperQuantizer
from core_pt.quantizers.ultra_quant import UltraQuantizer


class QuantizedKVCache(DynamicCache):
    """
    Hugging Face transformers 評価用カスタム KV キャッシュクラス
    指定したアルゴリズムで Key / Value テンソルを量子化・復元（劣化模倣）して格納します。
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
        self.quantizer: BaseQuantizer = None

    def _lazy_init_quantizer(self, device: torch.device, head_dim: int):
        """初回 update 呼び出し時にデバイスと次元に合わせて量子化器を動的初期化"""
        if self.quantizer is not None:
            # 既に初期化済みであっても、ヘッド次元が万が一異なるモデルに切り替わった場合に備えて更新
            if self.head_dim == head_dim:
                return

        self.head_dim = head_dim
        
        # デバイスタイプを動的に判定 ("cuda", "xpu", "cpu" などに対応)
        device_str = device.type

        if self.method == "fp16":
            self.quantizer = None
        elif self.method == "turbo_quant":
            self.quantizer = TurboQuantizer(
                head_dim=head_dim, num_bits=self.num_bits, device=device_str
            )
        elif self.method == "rope_aware_tq":
            self.quantizer = RoPEAwareTurboQuantizer(
                head_dim=head_dim,
                num_bands=2,
                bits_per_band=self.bits_per_band,
                device=device_str,
            )
        elif self.method == "hyper_quant":
            self.quantizer = HyperQuantizer(
                head_dim=head_dim, num_bits=self.num_bits, device=device_str
            )
        elif self.method == "ultra_quant":
            self.quantizer = UltraQuantizer(head_dim=head_dim, device=device_str)
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
        # 初回呼出時、あるいはモデル変更時に量子化器を動的初期化・追従
        self._lazy_init_quantizer(
            device=key_states.device, head_dim=key_states.shape[-1]
        )

        # FP16 (ベースライン) の場合は量子化スキップ
        if self.quantizer is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 1. KV キャッシュへの保存直前に圧縮＆デコードを適用
        key_dequant = self.quantizer.compress_and_decompress(key_states)
        value_dequant = self.quantizer.compress_and_decompress(value_states)

        # 2. 復元されたテンソルを親クラスのキャッシュ領域へ渡す
        return super().update(key_dequant, value_dequant, layer_idx, cache_kwargs)