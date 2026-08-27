import torch
import torch.nn as nn

from .base import BaseQuantizer


class RotorQuantizer(BaseQuantizer):
    """
    RotorQuant 簡易版（Simulated Quantization）。

    実装範囲:
      - Cl(3,0) ロータ（クォータニオン等価）による 3 次元ブロック回転
        （クォータニオン → 回転行列の式は標準的なもので正しい）
      - per-3D-block abs-max スケーリング + 一様量子化
    未実装:
      - ロータの最適化/キャリブレーション（本実装は seed 固定のランダム初期化のみ）
      - ビットパッキング

    注意: 3 次元ブロックごとに fp32 スケールを持つため、メタデータが
    量子化データ本体より大きくなる構造（解析的 footprint 参照）。

    head_dim が 3 で割り切れない場合の端数処理:
      head_dim=128 の場合、42 ブロック × 3 次元 = 126 次元のみを
      回転・量子化し、残余の 2 次元（tail）は回転も量子化もせず
      fp32 のまま生保持して連結する（compress の "tail" キー参照）。
      解析的 footprint のメタデータも (42 ブロック × 4B + 2 次元 × 4B)
      /トークン/ヘッド/KV で計算する（eval_pt/eval_compression.py 参照）。
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        device: str = "cuda",
        seed: int = 0,
    ):
        super().__init__(head_dim=head_dim, device=device)
        self.head_dim = head_dim
        self.bits = bits
        self.designed_bits = float(bits)
        self.block_size = 3  # Cl(3,0) に基づく3次元ブロック

        self.num_blocks = head_dim // self.block_size
        self.effective_dim = self.num_blocks * self.block_size

        # 再現性のため seed 固定（以前は run ごとに乱数が変わっていた）
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        init = torch.randn(self.num_blocks, 4, generator=g) * 0.02
        self.rotors_param = nn.Parameter(init.to(device))

        self.qmax = (2 ** (bits - 1)) - 1
        self.qmin = -(2 ** (bits - 1))

    def _get_rotor_matrices(self) -> torch.Tensor:
        """
        全ブロックのパラメータ [num_blocks, 4] から
        一括で [num_blocks, 3, 3] の回転行列を生成する（完全並列化）
        """
        norm = torch.norm(self.rotors_param, dim=-1, keepdim=True)
        p = self.rotors_param / (norm + 1e-8)

        s = p[:, 0]
        b1 = p[:, 1]
        b2 = p[:, 2]
        b3 = p[:, 3]

        # 3x3 回転行列を一括構築（クォータニオン→回転行列の標準式）
        R = torch.stack([
            torch.stack([1 - 2*(b2**2 + b3**2), 2*(b1*b2 - s*b3),     2*(b1*b3 + s*b2)], dim=-1),
            torch.stack([2*(b1*b2 + s*b3),     1 - 2*(b1**2 + b3**2), 2*(b2*b3 - s*b1)], dim=-1),
            torch.stack([2*(b1*b3 - s*b2),     2*(b2*b3 + s*b1),     1 - 2*(b1**2 + b2**2)], dim=-1)
        ], dim=-2)  # [num_blocks, 3, 3]

        return R

    def compress(self, tensor: torch.Tensor) -> dict:
        orig_dtype = tensor.dtype
        x = tensor.to(torch.float32)
        batch_shape = x.shape[:-1]

        main_part = x[..., :self.effective_dim]
        reshaped = main_part.view(*batch_shape, self.num_blocks, self.block_size)

        R = self._get_rotor_matrices().to(x.device)
        rotated = torch.einsum('...bi,bij->...bj', reshaped, R)

        scale = rotated.abs().max(dim=-1, keepdim=True).values / self.qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        quantized = torch.round(rotated / scale).clamp(self.qmin, self.qmax).to(torch.int8)

        return {
            "quantized_main": quantized,
            "scales": scale,
            "tail": x[..., self.effective_dim:] if self.effective_dim < self.head_dim else None,
            "dtype": orig_dtype
        }

    def decompress(self, compressed_data: dict) -> torch.Tensor:
        q_main = compressed_data["quantized_main"].to(torch.float32)
        scales = compressed_data["scales"]
        tail = compressed_data["tail"]
        orig_dtype = compressed_data["dtype"]
        batch_shape = q_main.shape[:-2]

        R = self._get_rotor_matrices().to(q_main.device)
        dequantized = q_main * scales

        recovered = torch.einsum('...bj,bij->...bi', dequantized, R)
        reconstructed_main = recovered.reshape(*batch_shape, self.effective_dim)

        if tail is not None:
            result = torch.cat([reconstructed_main, tail], dim=-1)
        else:
            result = reconstructed_main

        return result.to(orig_dtype)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(tensor))
