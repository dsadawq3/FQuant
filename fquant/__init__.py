"""
FQuant: High-Precision Post-Training LLM Quantization Framework by F-Labs.
Integrating DV-SSQ, KV-BSS, Walsh-Hadamard Spin, and Low-Rank Residual SVD (RCO).
"""

__version__ = "1.0.0"
__author__ = "F-Labs Team"

from .hadamard import generate_hadamard_matrix, apply_block_hadamard
from .gsq import quantize_gsq_int4, pack_int4_to_uint8, unpack_uint8_to_int4
from .rco_svd import compute_svd_residual_compensation
from .kv_bss import KVBSSAttentionHook, apply_kv_bss_attention
from .shield import ZeroCompressionShield
from .bifurcation import SpectralEntropyBifurcationDetector
from .engine import FQuantEngine

__all__ = [
    "generate_hadamard_matrix",
    "apply_block_hadamard",
    "quantize_gsq_int4",
    "pack_int4_to_uint8",
    "unpack_uint8_to_int4",
    "compute_svd_residual_compensation",
    "KVBSSAttentionHook",
    "apply_kv_bss_attention",
    "ZeroCompressionShield",
    "SpectralEntropyBifurcationDetector",
    "FQuantEngine",
]
