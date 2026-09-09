"""
Example: Basic 4-bit Group-Scale Quantization (GSQ) with Hadamard Spin and SVD RCO.
"""
import torch
from fquant.hadamard import get_hadamard_matrix, apply_hadamard_spin
from fquant.gsq import quantize_group_int4, dequantize_group_int4
from fquant.rco_svd import compute_svd_residual_compensation

def main():
    print("FQuant: Basic Layer Quantization Pipeline Demonstration")
    
    # 1. Synthesize layer weight: [M, N]
    M, N = 2048, 2048
    torch.manual_seed(42)
    weight = torch.randn(M, N, dtype=torch.bfloat16)
    
    # 2. Apply Hadamard spin
    print(f"Applying Hadamard spin to weight {tuple(weight.shape)}...")
    w_spin = apply_hadamard_spin(weight, block_size=128)
    
    # 3. Quantize to INT4 GSQ (group size 64)
    packed_q, scales = quantize_group_int4(w_spin, group_size=64)
    w_dequant = dequantize_group_int4(packed_q, scales, group_size=64)
    
    # 4. SVD Residual Compensation
    residual = w_spin - w_dequant
    rank = 16
    u_r, v_r, recon = compute_svd_residual_compensation(residual, rank=rank)
    
    total_reconstructed = w_dequant + recon
    final_error = torch.norm(w_spin - total_reconstructed).item()
    base_error = torch.norm(w_spin - w_dequant).item()
    
    print(f"Base INT4 reconstruction error: {base_error:.4f}")
    print(f"Error after SVD compensation (r={rank}): {final_error:.4f} (-{(1 - final_error/base_error)*100:.1f}%)")
    print("FQuant quantization roundtrip completed successfully.")

if __name__ == "__main__":
    main()
