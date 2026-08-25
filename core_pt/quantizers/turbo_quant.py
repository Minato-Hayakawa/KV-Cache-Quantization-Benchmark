import torch
from .base import BaseQuantizer


class TurboQuantizer(BaseQuantizer):
    """
    TurboQuant 簡易版（Simulated Quantization）。

    実装範囲:
      - ランダム直交回転 R (seed 固定で再現性確保)
      - per-token abs-max スケーリング + 一様スカラー量子化 (num_bits)
    未実装:
      - PolarQuant（極座標分解）、QJL 1ビット残差補正、ビットパッキング

    保存形式はシミュレーションのため int8 テンソル + fp32 スケールであり、
    実メモリ占有は designed_bits を反映しない（解析的 footprint 参照）。
    """

    def __init__(self, head_dim: int = 128, num_bits: int = 3, device: str = "cuda", seed: int = 0):
        super().__init__(head_dim=head_dim, device=device)
        self.num_bits = num_bits
        self.designed_bits = float(num_bits)

        # 再現性のため seed 固定（以前は run ごとに乱数が変わっていた）
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        random_matrix = torch.randn(head_dim, head_dim, generator=g)
        q, _ = torch.linalg.qr(random_matrix)
        self.R = q.to(device)

    def compress(self, tensor: torch.Tensor) -> dict:
        tensor_f32 = tensor.to(torch.float32)
        rotated = torch.matmul(tensor_f32, self.R.to(tensor.device))

        qmax = (2 ** (self.num_bits - 1)) - 1
        qmin = -(2 ** (self.num_bits - 1))
        scale = rotated.abs().max(dim=-1, keepdim=True).values / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        quantized = torch.round(rotated / scale).clamp(qmin, qmax).to(torch.int8)

        return {"quantized": quantized, "scale": scale, "dtype": tensor.dtype}

    def decompress(self, compressed_data: dict) -> torch.Tensor:
        quantized = compressed_data["quantized"].to(torch.float32)
        scale = compressed_data["scale"]
        orig_dtype = compressed_data["dtype"]

        dequantized = quantized * scale
        recovered = torch.matmul(dequantized, self.R.T.to(dequantized.device))
        return recovered.to(orig_dtype)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(tensor))
