"""
Group-Scale INT4 Quantization and Bit-Packing Utilities.
"""

import torch

def quantize_gsq_int4(weight: torch.Tensor, group_size: int = 64):
    """
    Quantizes a 2D weight matrix [M, N] into INT4 values (-8 to 7) using per-group scaling.
    Returns:
        q_weight: int8 tensor of quantized values
        scale: per-group scale tensor [M, N // group_size]
        w_dequant: reconstructed baseline weight
    """
    m, n = weight.shape
    assert n % group_size == 0, f"Dimension n={n} not divisible by group_size={group_size}"
    
    w_grouped = weight.float().view(m, n // group_size, group_size)
    max_abs = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True).clamp(min=1e-5)
    scale = max_abs / 7.0
    
    q_grouped = torch.clamp(torch.round(w_grouped / scale), -8, 7).to(torch.int8)
    w_dequant = (q_grouped.float() * scale).view(m, n)
    
    return q_grouped.view(m, n), scale.squeeze(-1), w_dequant

def pack_int4_to_uint8(q_int8: torch.Tensor) -> torch.Tensor:
    """
    Packs two signed INT4 values (-8 to 7) into a single unsigned UINT8 byte.
    Low nibble = q[:, 0::2], High nibble = q[:, 1::2].
    """
    assert q_int8.shape[-1] % 2 == 0, "Last dimension must be even for 4-bit packing"
    low = (q_int8[..., 0::2] & 0x0F).to(torch.uint8)
    high = ((q_int8[..., 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = (low | high).contiguous()
    return packed

def unpack_uint8_to_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpacks packed UINT8 bytes into signed INT4 values (-8 to 7) stored as torch.int8.
    """
    low = (packed & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high)
    
    orig_shape = list(packed.shape)
    orig_shape[-1] = orig_shape[-1] * 2
    unpacked = torch.empty(orig_shape, dtype=torch.int8, device=packed.device)
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    return unpacked
