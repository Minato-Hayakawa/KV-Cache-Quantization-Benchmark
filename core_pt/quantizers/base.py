import abc
import torch


class BaseQuantizer(abc.ABC):
    """
    KV Cache Quantizer Base Class
    """

    def __init__(self, head_dim: int = 128, device: str = "cuda"):
        self.head_dim = head_dim
        self.device = device
        # 実装が実際に使う論理ビット幅（実装レベル数から導出する。手書き定数禁止）
        self.designed_bits = None

    @abc.abstractmethod
    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch_size, num_heads, seq_len, head_dim)
        Output shape: (batch_size, num_heads, seq_len, head_dim) (Quantized & Dequantized tensor)
        """
        pass