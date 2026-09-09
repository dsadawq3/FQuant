"""
Zero-Compression Shield.
Identifies and preserves scale-critical parameters (RMSNorms, Biases, Embeddings) in pristine BF16/FP16.
"""

import torch

class ZeroCompressionShield:
    @staticmethod
    def is_shielded(param_name: str) -> bool:
        name_lower = param_name.lower()
        return (
            "norm" in name_lower
            or "layernorm" in name_lower
            or "bias" in name_lower
            or "embed_tokens" in name_lower
            or "lm_head" in name_lower
        )

    @staticmethod
    def protect(tensor: torch.Tensor, target_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return tensor.to(target_dtype).contiguous()
