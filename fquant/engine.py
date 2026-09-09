"""
FQuantEngine: Universal Multi-Tier Quantization Pipeline.
"""

import torch
from .hadamard import apply_block_hadamard
from .gsq import quantize_gsq_int4, pack_int4_to_uint8
from .rco_svd import compute_svd_residual_compensation
from .shield import ZeroCompressionShield
from .bifurcation import SpectralEntropyBifurcationDetector

class FQuantEngine:
    def __init__(self, group_size: int = 64, default_rank: int = 16, k_proj_rank: int = 32):
        self.group_size = group_size
        self.default_rank = default_rank
        self.k_proj_rank = k_proj_rank

    def quantize_parameter(self, name: str, tensor: torch.Tensor, layer_idx: int = None, total_layers: int = 32):
        if ZeroCompressionShield.is_shielded(name):
            return {
                "type": "shield",
                "tensor": ZeroCompressionShield.protect(tensor)
            }
            
        is_k_proj = "k_proj" in name
        rank = SpectralEntropyBifurcationDetector.get_layer_rank(
            layer_idx=layer_idx if layer_idx is not None else 0,
            total_layers=total_layers,
            is_k_proj=is_k_proj,
            default_rank=self.default_rank
        )
        
        # 1. Walsh-Hadamard Spin Rotation
        w = tensor.clone()
        m, n = w.shape
        if n % 128 == 0:
            w_rotated = apply_block_hadamard(w, block_size=128, dim=-1)
        else:
            w_rotated = w
            
        # 2. INT4 GSQ
        q_w, scales, w_dequant = quantize_gsq_int4(w_rotated, group_size=self.group_size)
        packed_q = pack_int4_to_uint8(q_w)
        
        # 3. Residual SVD Compensation (RCO)
        residual = w_rotated.float() - w_dequant
        factor_a, factor_b, _ = compute_svd_residual_compensation(residual, rank=rank)
        
        return {
            "type": "quantized",
            "qweight_packed": packed_q,
            "scales": scales.to(torch.bfloat16).contiguous(),
            "svd_a": factor_a,
            "svd_b": factor_b,
            "rank": rank
        }
