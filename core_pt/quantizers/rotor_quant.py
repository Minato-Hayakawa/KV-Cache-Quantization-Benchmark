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
        
        # ヘッド次元が3で割り切れる前提、余りはそのまま通す
        self.num_blocks = head_dim // self.block_size
        self.effective_dim = self.num_blocks * self.block_size
        
        # 各3Dブロック用のローターパラメータ（初期化時は単位元付近またはランダムな正規化クォータニオン/ローター表現）
        # Cl(3,0)のローターは 4 パラメータ（scalar 1つ + bivector 3つ）
        self.rotors_param = nn.Parameter(
            torch.randn(self.num_blocks, 4, device=device) * 0.02
        )
        
        self.qmax = (2 ** (bits - 1)) - 1
        self.qmin = -(2 ** (bits - 1))

    def _get_rotor_matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        4つのローターパラメータから 3x3 の直交回転行列（サンドイッチ積用）を構築する
        param: [4] (scalar, b12, b23, b31)
        """
        # 正規化して単位ローター（SO(3)回転）にする
        norm = torch.norm(param)
        if norm > 0:
        # パラメータを正規化して安定化
            p = param / (norm + 1e-8)
        else:
            p = torch.tensor([1.0, 0.0, 0.0, 0.0], device=param.device)
            
        s, b1, b2, b3 = p[0], p[1], p[2], p[3]
        
        # 四元数 / ローターの成分から 3x3 回転行列 R を生成
        # R = I + 2s(B) + 2(B^2) の関係から展開
        R = torch.stack([
            torch.stack([1 - 2*(b2**2 + b3**2), 2*(b1*b2 - s*b3),     2*(b1*b3 + s*b2)]),
            torch.stack([2*(b1*b2 + s*b3),     1 - 2*(b1**2 + b3**2), 2*(b2*b3 - s*b1)]),
            torch.stack([2*(b1*b3 - s*b2),     2*(b2*b3 + s*b1),     1 - 2*(b1**2 + b2**2)])
        ])
        return R

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        tensor: [..., head_dim] (例: [Batch, SeqLen, 128] などの Key テンソル)
        """
        orig_dtype = tensor.dtype
        x = tensor.to(torch.float32)
        
        batch_shape = x.shape[:-1]
        
        # 1. 有効次元 (3の倍数) と端数に分割
        main_part = x[..., :self.effective_dim]
        # [..., num_blocks, 3] に変形
        reshaped = main_part.view(*batch_shape, self.num_blocks, self.block_size)
        
        processed_blocks = []
        
        # 2. ブロックごとにローター回転 ＆ 量子化
        for i in range(self.num_blocks):
            block = reshaped[..., i, :]  # [..., 3]
            R = self._get_rotor_matrix(self.rotors_param[i])  # [3, 3]
            
            # --- 前方回転 (Forward Rotation: R * v) ---
            # 形状合わせのため matmul: [..., 3] @ [3, 3].T または [3, 3] @ [..., 3, 1]
            rotated = torch.matmul(block, R.t())
            
            # --- 量子化 ＆ 逆量子化 (Lloyd-Max / 均一スケール近似) ---
            scale = rotated.abs().max(dim=-1, keepdim=True).values / self.qmax
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            
            quantized = torch.round(rotated / scale).clamp(self.qmin, self.qmax)
            dequantized = quantized * scale
            
            # --- 逆回転 (Inverse Rotation: R^T * v または 逆ローター) ---
            recovered = torch.matmul(dequantized, R)  # R.t() の転置は R
            processed_blocks.append(recovered)
            
        # 3. 再結合
        reconstructed_main = torch.stack(processed_blocks, dim=-2).view(*batch_shape, self.effective_dim)
        
        if self.effective_dim < self.head_dim:
            tail_part = x[..., self.effective_dim:]
            result = torch.cat([reconstructed_main, tail_part], dim=-1)
        else:
            result = reconstructed_main
            
        return result.to(orig_dtype)
    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        x = tensor.to(torch.float32)
        batch_shape = x.shape[:-1]
        
        main_part = x[..., :self.effective_dim]
        reshaped = main_part.view(*batch_shape, self.num_blocks, self.block_size)
        
        quantized_blocks = []
        scales = []
        
        for i in range(self.num_blocks):
            block = reshaped[..., i, :]
            R = self._get_rotor_matrix(self.rotors_param[i])
            rotated = torch.matmul(block, R.t())
            
            scale = rotated.abs().max(dim=-1, keepdim=True).values / self.qmax
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            
            quantized = torch.round(rotated / scale).clamp(self.qmin, self.qmax).to(torch.int8)
            quantized_blocks.append(quantized)
            scales.append(scale)
            
        reconstructed_q = torch.stack(quantized_blocks, dim=-2)
        stacked_scales = torch.stack(scales, dim=-3) # 簡略化のための保持
        
        return {
            "quantized_main": reconstructed_q,
            "scales": stacked_scales,
            "tail": x[..., self.effective_dim:] if self.effective_dim < self.head_dim else None,
            "dtype": orig_dtype
        }

    def decompress(self, compressed_data: dict) -> torch.Tensor:
        q_main = compressed_data["quantized_main"].to(torch.float32)
        scales = compressed_data["scales"]
        tail = compressed_data["tail"]
        orig_dtype = compressed_data["dtype"]
        batch_shape = q_main.shape[:-2]
        
        processed_blocks = []
        for i in range(self.num_blocks):
            block = q_main[..., i, :]
            scale = scales[..., i, :, :]
            R = self._get_rotor_matrix(self.rotors_param[i])
            
            dequantized = block * scale
            recovered = torch.matmul(dequantized, R)
            processed_blocks.append(recovered)
            
        reconstructed_main = torch.stack(processed_blocks, dim=-2).view(*batch_shape, self.effective_dim)
        
        if tail is not None:
            result = torch.cat([reconstructed_main, tail], dim=-1)
        else:
            result = reconstructed_main
            
        return result.to(orig_dtype)

    def compress_and_decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(tensor))