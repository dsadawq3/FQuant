"""
Walsh-Hadamard Matrix Generation and Block-Diagonal Spin Rotation.
Suppresses activation outliers by rotating coordinates into an isotropic distribution.
"""

import math
import torch

_HADAMARD_CACHE = {}

def generate_hadamard_matrix(n: int, dtype: torch.dtype = torch.float32, device: torch.device = None) -> torch.Tensor:
    """
    Generates an orthonormal Sylvester Walsh-Hadamard matrix H_n such that H^T H = I.
    n must be a power of 2.
    """
    assert (n & (n - 1)) == 0 and n > 0, f"Dimension n={n} must be a power of 2"
    
    key = (n, dtype, str(device))
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]
        
    if n == 1:
        h = torch.tensor([[1.0]], dtype=dtype, device=device)
    else:
        h_half = generate_hadamard_matrix(n // 2, dtype=dtype, device=device)
        top = torch.cat([h_half, h_half], dim=1)
        bottom = torch.cat([h_half, -h_half], dim=1)
        h = torch.cat([top, bottom], dim=0) / math.sqrt(2.0)
        
    _HADAMARD_CACHE[key] = h
    return h

def apply_block_hadamard(tensor: torch.Tensor, block_size: int = 128, dim: int = -1) -> torch.Tensor:
    """
    Applies block-diagonal Walsh-Hadamard rotation along the specified dimension.
    The size of tensor along `dim` must be divisible by `block_size`.
    """
    orig_shape = tensor.shape
    d = orig_shape[dim]
    assert d % block_size == 0, f"Dimension {d} is not divisible by block_size={block_size}"
    
    h = generate_hadamard_matrix(block_size, dtype=tensor.dtype, device=tensor.device)
    
    if dim == -1 or dim == len(orig_shape) - 1:
        reshaped = tensor.view(-1, d // block_size, block_size)
        rotated = torch.matmul(reshaped, h)
        return rotated.view(orig_shape)
    elif dim == 0:
        reshaped = tensor.view(d // block_size, block_size, -1)
        rotated = torch.matmul(h.t(), reshaped)
        return rotated.view(orig_shape)
    else:
        raise NotImplementedError(f"Block Hadamard along dim {dim} not supported")
