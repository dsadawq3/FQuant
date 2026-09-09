"""
SVD Residual Compensation (SRC) via Truncated Singular Value Decomposition (SVD).
Recovers the high-curvature eigenspace lost during 4-bit discretization.
"""

import torch

def compute_svd_residual_compensation(residual: torch.Tensor, rank: int = 16):
    """
    Computes truncated SVD on residual error matrix:
        R approx U_r * Sigma_r * V_r^T = A * B
    Where:
        A = U_r * sqrt(Sigma_r) in R^{M x r}
        B = sqrt(Sigma_r) * V_r^T in R^{r x N}
    """
    m, n = residual.shape
    try:
        u, s, vh = torch.linalg.svd(residual.float(), full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        
        sqrt_s = torch.sqrt(s_r.clamp(min=1e-7))
        factor_a = (u_r * sqrt_s.unsqueeze(0)).to(torch.bfloat16).contiguous()
        factor_b = (sqrt_s.unsqueeze(1) * vh_r).to(torch.bfloat16).contiguous()
        recon_residual = torch.matmul(factor_a.float(), factor_b.float())
    except Exception as e:
        factor_a = torch.zeros((m, rank), dtype=torch.bfloat16)
        factor_b = torch.zeros((rank, n), dtype=torch.bfloat16)
        recon_residual = torch.zeros_like(residual)
        
    return factor_a, factor_b, recon_residual
