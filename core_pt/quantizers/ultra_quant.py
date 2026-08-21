import torch
import math
from .base import BaseQuantizer


class UltraQuantizer(BaseQuantizer):
    """
    UltraQuant: Walsh-Hadamard 変換 ＋ FP4 (E2M1) ダイレクトマッピング
    """

    # FP4 (E2M1) 標準表現グリッド (16点)
    E2M1_GRID = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ]

    def __init__(self, head_dim: int = 128, device: str = "cuda"):
        super().__init__(head_dim=head_dim, device=device)
        self.grid = torch.tensor(self.E2M1_GRID, device=device)
        self.hadamard_matrix = self._generate_hadamard(head_dim).to(device)

    def _generate_hadamard(self, n: int) -> torch.Tensor:
        if n == 1:
            return torch.tensor([[1.0]])
        H_sub = self._generate_hadamard(n // 2)
        top = torch.cat([H_sub, H_sub], dim=1)
        bottom = torch.cat([H_sub, -H_sub], dim=1)
        H = torch.cat([top, bottom], dim=0)
        return H / math.sqrt(2.0)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)

        # 1. Walsh-Hadamard 変換
        H = self.hadamard_matrix.to(tensor.device)
        transformed = torch.matmul(tensor_f32, H)

        # 2. FP4 (E2M1) スケーリング (最大値 6.0)
        max_val = transformed.abs().max(dim=-1, keepdim=True).values
        scale = max_val / 6.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        scaled_val = transformed / scale

        # 3. FP4 グリッドへの最近傍マッピング (Nearest Neighbor)
        grid = self.grid.to(tensor.device)
        # (..., head_dim, 1) - (16,) -> 最近傍インデックスを探索
        diffs = (scaled_val.unsqueeze(-1) - grid).abs()
        nearest_idx = diffs.argmin(dim=-1)

        dequantized = grid[nearest_idx] * scale

        # 4. 逆 FWHT
        recovered = torch.matmul(dequantized, H.T)
        return recovered.to(orig_dtype)
    def compress(self, tensor: torch.Tensor) -> dict:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)

        # 1. Walsh-Hadamard 変換
        H = self.hadamard_matrix.to(tensor.device)
        transformed = torch.matmul(tensor_f32, H)

        # 2. FP4 (E2M1) スケーリング
        max_val = transformed.abs().max(dim=-1, keepdim=True).values
        scale = max_val / 6.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        scaled_val = transformed / scale
        grid = self.grid.to(tensor.device)
        diffs = (scaled_val.unsqueeze(-1) - grid).abs()
        nearest_idx = diffs.argmin(dim=-1).to(torch.int8) # インデックスを保存（実質4bit/8bit）

        return {
            "quantized_idx": nearest_idx,
            "scale": scale,
            "dtype": orig_dtype
        }

    def decompress(self, compressed_data: dict) -> torch.Tensor:
        nearest_idx = compressed_data["quantized_idx"].long()
        scale = compressed_data["scale"]
        orig_dtype = compressed_data["dtype"]
        
        grid = self.grid.to(scale.device)
        dequantized = grid[nearest_idx] * scale

        H = self.hadamard_matrix.to(dequantized.device)
        recovered = torch.matmul(dequantized, H.T)
        return recovered.to(orig_dtype)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(tensor))