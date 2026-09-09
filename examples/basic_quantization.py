"""
Example: Basic 4-bit Group-Scale Quantization (GSQ) with Hadamard Spin and SVD RCO.
"""
import torch
from fquant.hadamard import generate_hadamard_matrix, apply_block_hadamard
from fquant.gsq import quantize_gsq_int4, pack_int4_to_uint8, unpack_uint8_to_int4
from fquant.rco_svd import compute_svd_residual_compensation


def main():
    print("FQuant: Basic Layer Quantization Pipeline Demonstration")

    # 1. Synthesize layer weight: [M, N]
    M, N = 256, 2048
    torch.manual_seed(42)
    weight = torch.randn(M, N, dtype=torch.bfloat16)

    # Show Hadamard matrix generation works
    H = generate_hadamard_matrix(128)
    print(f"Hadamard H128 orthonormal check: {(H @ H.T).diagonal().mean().item():.4f} (expect ~1.0)")

    # 2. Apply Hadamard spin (N=2048 divisible by block 128)
    print(f"Applying Hadamard spin to weight {tuple(weight.shape)}...")
    w_spin = apply_block_hadamard(weight, block_size=128, dim=-1)

    # 3. Quantize to INT4 GSQ (group size 64)
    q_w, scales, w_dequant = quantize_gsq_int4(w_spin, group_size=64)
    packed_q = pack_int4_to_uint8(q_w)
    unpacked_q = unpack_uint8_to_int4(packed_q)
    assert torch.equal(q_w.cpu(), unpacked_q.cpu()), "pack/unpack roundtrip mismatch"
    print(f"Packed INT4: {tuple(packed_q.shape)}, scales: {tuple(scales.shape)} (pack/unpack OK)")

    # 4. SVD Residual Compensation: returns (factor_a, factor_b, recon)
    residual = w_spin.float() - w_dequant
    rank = 16
    factor_a, factor_b, recon = compute_svd_residual_compensation(residual, rank=rank)

    total_reconstructed = w_dequant + recon
    final_error = torch.norm(w_spin.float() - total_reconstructed.float()).item()
    base_error = torch.norm(w_spin.float() - w_dequant).item()

    print(f"Base INT4 reconstruction error: {base_error:.4f}")
    print(f"Error after SVD compensation (r={rank}): {final_error:.4f} (-{(1 - final_error / max(base_error, 1e-9)) * 100:.1f}%)")
    print("FQuant quantization roundtrip completed successfully.")


if __name__ == "__main__":
    main()
