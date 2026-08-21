import torch
import math
from .base import BaseQuantizer


class HyperQuantizer(BaseQuantizer):
    """
    HyperQuant: アダマール変換 (RHT) ＋ D4 格子 (Lattice) 近似射影
    """

    def __init__(self, head_dim: int = 128, num_bits: int = 3, device: str = "cuda"):
        super().__init__(head_dim=head_dim, device=device)
        self.num_bits = num_bits
        self.hadamard_matrix = self._generate_hadamard(head_dim).to(device)

    def _generate_hadamard(self, n: int) -> torch.Tensor:
        """n次元アダマール行列の生成 (nは2の累乗)"""
        if n == 1:
            return torch.tensor([[1.0]], device=self.device)
        H_sub = self._generate_hadamard(n // 2)
        top = torch.cat([H_sub, H_sub], dim=1)
        bottom = torch.cat([H_sub, -H_sub], dim=1)
        H = torch.cat([top, bottom], dim=0)
        return H / math.sqrt(2.0)

    def _project_d4_lattice(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """D4 格子 (4次元単位で要素和が偶数となる格子点) への射影 (ベクトル化高速版)"""
        scaled = tensor / scale
        rounded = torch.round(scaled)

        # 4次元ごとに分割
        shape = rounded.shape
        reshaped = rounded.view(-1, 4)
        scaled_4 = scaled.view(-1, 4)
        
        sums = reshaped.sum(dim=-1)

        # 和が奇数の箇所を検知して補正
        odd_mask = (sums % 2 != 0)
        if odd_mask.any():
            diffs = (scaled_4 - reshaped).abs()
            max_idx = diffs.argmax(dim=-1)  # (N,)
            
            # ループを使わず、PyTorchのインデクシングで一括処理
            row_indices = torch.where(odd_mask)[0]
            col_indices = max_idx[row_indices]
            
            diff_vals = scaled_4[row_indices, col_indices] - reshaped[row_indices, col_indices]
            adjustment = torch.where(diff_vals >= 0, 1.0, -1.0)
            
            reshaped[row_indices, col_indices] += adjustment

        return reshaped.view(shape) * scale

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)

        # 1. RHT (ランダム・アダマール変換)
        H = self.hadamard_matrix.to(tensor.device)
        transformed = torch.matmul(tensor_f32, H)

        # 2. 格子量子化
        qmax = (2 ** (self.num_bits - 1)) - 1
        scale = transformed.abs().max(dim=-1, keepdim=True).values / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        dequantized = self._project_d4_lattice(transformed, scale)

        # 3. 逆 RHT
        recovered = torch.matmul(dequantized, H.T)
        return recovered.to(orig_dtype)
    def compress(self, tensor: torch.Tensor) -> dict:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)

        # 1. RHT (ランダム・アダマール変換)
        H = self.hadamard_matrix.to(tensor.device)
        transformed = torch.matmul(tensor_f32, H)

        # 2. 格子量子化
        qmax = (2 ** (self.num_bits - 1)) - 1
        scale = transformed.abs().max(dim=-1, keepdim=True).values / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        # _project_d4_lattice の中で丸められたものを int8 にキャストして保存
        scaled = transformed / scale
        rounded = torch.round(scaled).to(torch.int8)

        return {
            "quantized": rounded,
            "scale": scale,
            "dtype": orig_dtype
        }

    def decompress(self, compressed_data: dict) -> torch.Tensor:
        quantized = compressed_data["quantized"].to(torch.float32)
        scale = compressed_data["scale"]
        orig_dtype = compressed_data["dtype"]
        
        # D4格子の近似復元 ＋ 逆 RHT
        dequantized = quantized * scale
        # 必要に応じて厳密なD4射影を通すか、そのままスケーリング復元
        H = self.hadamard_matrix.to(dequantized.device)
        recovered = torch.matmul(dequantized, H.T)
        return recovered.to(orig_dtype)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(tensor))