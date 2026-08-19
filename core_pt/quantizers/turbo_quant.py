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


class TurboQuantizer(BaseQuantizer):
    """
    TurboQuant: 全体直交回転 ＋ 均一コードブック量子化
    """

    def __init__(self, head_dim: int = 128, num_bits: int = 3, device: str = "cuda"):
        super().__init__(head_dim=head_dim, device=device)
        self.num_bits = num_bits

        # QR分解によりランダム直交回転行列 R を生成 (d x d)
        random_matrix = torch.randn(head_dim, head_dim, device=device)
        q, _ = torch.linalg.qr(random_matrix)
        self.R = q

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)

        # 1. 直交回転: rotated = X @ R
        rotated = torch.matmul(tensor_f32, self.R.to(tensor.device))

        # 2. 均一スカラー量子化
        qmax = (2 ** (self.num_bits - 1)) - 1
        qmin = -(2 ** (self.num_bits - 1))

        scale = rotated.abs().max(dim=-1, keepdim=True).values / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        quantized = torch.round(rotated / scale).clamp(qmin, qmax)
        dequantized = quantized * scale

        # 3. 逆直交回転: recovered = rotated_deq @ R^T
        recovered = torch.matmul(dequantized, self.R.T.to(tensor.device))
        return recovered.to(orig_dtype)