import sys
import os

# core_pt フォルダへのパスを動的に追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core_pt')))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_cache import QuantizedKVCache  # これで core_pt から正常にインポートできる
import torch
from .base import BaseQuantizer


class RoPEAwareTurboQuantizer(BaseQuantizer):
    """
    RoPE-Aware TurboQuant: 周波数帯域別分割回転 ＋ 動的ビット配分 (例: 高周波 4-bit / 低周波 2-bit)
    """

    def __init__(
        self,
        head_dim: int = 128,
        num_bands: int = 2,
        bits_per_band: list = None,
        device: str = "cuda",
    ):
        super().__init__(head_dim=head_dim, device=device)
        self.num_bands = num_bands
        self.bits_per_band = bits_per_band or [4, 2]  # 平均 3-bit
        self.band_dim = head_dim // num_bands

        # 帯域ごとの小直交行列 (R_high, R_low 等)
        self.rotation_matrices = []
        for _ in range(num_bands):
            rand_m = torch.randn(self.band_dim, self.band_dim, device=device)
            q, _ = torch.linalg.qr(rand_m)
            self.rotation_matrices.append(q)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)
        output_bands = []

        for i in range(self.num_bands):
            start_idx = i * self.band_dim
            end_idx = (i + 1) * self.band_dim

            # 1. 帯域スライス
            band = tensor_f32[..., start_idx:end_idx]

            # 2. 帯域内直交回転
            R = self.rotation_matrices[i].to(tensor.device)
            rotated = torch.matmul(band, R)

            # 3. 周波数適応型量子化
            bits = self.bits_per_band[i]
            qmax = (2 ** (bits - 1)) - 1
            qmin = -(2 ** (bits - 1))

            scale = rotated.abs().max(dim=-1, keepdim=True).values / qmax
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)

            quantized = torch.round(rotated / scale).clamp(qmin, qmax)
            dequantized = quantized * scale

            # 4. 逆回転
            recovered_band = torch.matmul(dequantized, R.T)
            output_bands.append(recovered_band)

        # 全帯域を結合
        result = torch.cat(output_bands, dim=-1)
        return result.to(orig_dtype)