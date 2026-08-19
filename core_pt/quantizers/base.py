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
import abc
import torch


class BaseQuantizer(abc.ABC):
    """
    KV Cache Quantizer Base Class
    """

    def __init__(self, head_dim: int = 128, device: str = "cuda"):
        self.head_dim = head_dim
        self.device = device

    @abc.abstractmethod
    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch_size, num_heads, seq_len, head_dim)
        Output shape: (batch_size, num_heads, seq_len, head_dim) (Quantized & Dequantized tensor)
        """
        pass