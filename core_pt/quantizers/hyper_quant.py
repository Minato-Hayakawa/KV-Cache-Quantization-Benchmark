import torch
import math
from .base import BaseQuantizer


class HyperQuantizer(BaseQuantizer):
    """
    HyperQuant 簡易版（Simulated Quantization）。

    実装範囲:
      - 直交 Hadamard 変換 (RHT のうち回転部分)
      - 4 次元単位の D4 格子（成分和が偶数の整数格子）射影
        ※ 以前は compress() 経路に適用されず、キャッシュ経由では
           「Hadamard + 素の丸め」に退化していたバグを修正
      - per-token abs-max スケーリング
    未実装:
      - Rice エントロピー符号化（これが無いと論文の 1.7–2 bps という
        レート主張は成立しない。本実装で達成可能なビット幅は num_bits）
      - ビットパッキング
    """

    def __init__(self, head_dim: int = 128, num_bits: int = 3, device: str = "cuda"):
        super().__init__(head_dim=head_dim, device=device)
        self.num_bits = num_bits
        self.designed_bits = float(num_bits)
        self.hadamard_matrix = self._generate_hadamard(head_dim).to(device)

    def _generate_hadamard(self, n: int) -> torch.Tensor:
        """n次元アダマール行列の生成 (nは2の累乗、直交行列になるよう正規化済み)"""
        if n == 1:
            return torch.tensor([[1.0]], device=self.device)
        H_sub = self._generate_hadamard(n // 2)
        top = torch.cat([H_sub, H_sub], dim=1)
        bottom = torch.cat([H_sub, -H_sub], dim=1)
        H = torch.cat([top, bottom], dim=0)
        return H / math.sqrt(2.0)

    def _d4_lattice_round(self, scaled: torch.Tensor) -> torch.Tensor:
        """
        スカラ丸め後の整数値を D4 格子（4次元グループ内で成分和が偶数となる格子）
        へ射影する。各グループで和が奇数の場合、丸め誤差が最大の成分を ±1 補正する。
        head_dim が 4 の倍数でない尻尾部分は素の丸めのまま残す。
        """
        rounded = torch.round(scaled)
        d = scaled.shape[-1]
        main = d - (d % 4)
        if main <= 0:
            return rounded

        r_main = rounded[..., :main].reshape(-1, 4)
        s_main = scaled[..., :main].reshape(-1, 4)

        sums = r_main.sum(dim=-1)
        odd_mask = (sums % 2 != 0)
        if odd_mask.any():
            diffs = (s_main - r_main).abs()
            max_idx = diffs.argmax(dim=-1)

            rows = torch.where(odd_mask)[0]
            cols = max_idx[rows]
            direction = torch.where(
                (s_main[rows, cols] - r_main[rows, cols]) >= 0, 1.0, -1.0
            )
            r_main[rows, cols] += direction

        corrected = torch.cat(
            [r_main.view(*scaled.shape[:-1], main), rounded[..., main:]], dim=-1
        )
        return corrected

    def compress(self, tensor: torch.Tensor) -> dict:
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.to(torch.float32)

        H = self.hadamard_matrix.to(tensor.device)
        transformed = torch.matmul(tensor_f32, H)

        qmax = (2 ** (self.num_bits - 1)) - 1
        qmin = -(2 ** (self.num_bits - 1))
        scale = transformed.abs().max(dim=-1, keepdim=True).values / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        scaled = transformed / scale
        # 【修正】D4 格子射影を実際の保存経路にも適用
        int_q = self._d4_lattice_round(scaled).clamp(qmin, qmax).to(torch.int8)

        return {
            "quantized": int_q,
            "scale": scale,
            "dtype": orig_dtype
        }

    def decompress(self, compressed_data: dict) -> torch.Tensor:
        quantized = compressed_data["quantized"].to(torch.float32)
        scale = compressed_data["scale"]
        orig_dtype = compressed_data["dtype"]

        dequantized = quantized * scale
        H = self.hadamard_matrix.to(dequantized.device)
        recovered = torch.matmul(dequantized, H.T)
        return recovered.to(orig_dtype)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(tensor))
