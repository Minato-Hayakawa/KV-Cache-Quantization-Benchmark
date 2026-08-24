import torch
import torch.nn as nn

class RotorQuantizer(nn.Module):
    """
    RotorQuant (Cl(3,0) Clifford Algebra Rotor-based KV Cache Quantizer)
    - 3次元ブロックごとのローター回転 (サンドイッチ積) によりデカルト相関を解消
    - 従来のグローバル直交行列（d x d）を排除し、O(d) の計算量と劇的なパラメータ削減を実現
    """
    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        device: str = "cuda"
    ):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.block_size = 3  # Cl(3,0) に基づく3次元ブロック
        
        self.num_blocks = head_dim // self.block_size
        self.effective_dim = self.num_blocks * self.block_size
        
        # 各3Dブロック用のローターパラメータ
        self.rotors_param = nn.Parameter(
            torch.randn(self.num_blocks, 4, device=device) * 0.02
        )
        
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
        
        # 3x3 回転行列を一括構築
        R = torch.stack([
            torch.stack([1 - 2*(b2**2 + b3**2), 2*(b1*b2 - s*b3),     2*(b1*b3 + s*b2)], dim=-1),
            torch.stack([2*(b1*b2 + s*b3),     1 - 2*(b1**2 + b3**2), 2*(b2*b3 - s*b1)], dim=-1),
            torch.stack([2*(b1*b3 - s*b2),     2*(b2*b3 + s*b1),     1 - 2*(b1**2 + b2**2)], dim=-1)
        ], dim=-2)  # [num_blocks, 3, 3]
        
        return R

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        x = tensor.to(torch.float32)
        batch_shape = x.shape[:-1]
        
        main_part = x[..., :self.effective_dim]
        reshaped = main_part.view(*batch_shape, self.num_blocks, self.block_size)
        
        # 一括で回転行列を取得 [num_blocks, 3, 3]
        R = self._get_rotor_matrices().to(x.device)
        
        # --- 前方回転 (一括アインシュタイン縮約) ---
        rotated = torch.einsum('...bi,bij->...bj', reshaped, R)
        
        # --- 量子化 ＆ 逆量子化 ---
        scale = rotated.abs().max(dim=-1, keepdim=True).values / self.qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        quantized = torch.round(rotated / scale).clamp(self.qmin, self.qmax)
        dequantized = quantized * scale
        
        # --- 逆回転 ---
        recovered = torch.einsum('...bj,bij->...bi', dequantized, R)
        
        reconstructed_main = recovered.reshape(*batch_shape, self.effective_dim)
        
        if self.effective_dim < self.head_dim:
            tail_part = x[..., self.effective_dim:]
            result = torch.cat([reconstructed_main, tail_part], dim=-1)
        else:
            result = reconstructed_main
            
        return result.to(orig_dtype)

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