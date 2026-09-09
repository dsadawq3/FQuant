import unittest
import torch
from fquant.hadamard import generate_hadamard_matrix, apply_block_hadamard

class TestHadamard(unittest.TestCase):
    def test_orthonormality(self):
        for n in [16, 64, 128]:
            h = generate_hadamard_matrix(n)
            prod = torch.matmul(h, h.t())
            eye = torch.eye(n)
            diff = torch.norm(prod - eye).item()
            self.assertLess(diff, 1e-4, f"Hadamard matrix H_{n} is not orthonormal, diff={diff}")

    def test_dot_product_preservation(self):
        d = 256
        x = torch.randn(4, d)
        w = torch.randn(128, d)
        orig_out = torch.matmul(x, w.t())

        # Rotate x and w
        x_rot = apply_block_hadamard(x, block_size=128, dim=-1)
        w_rot = apply_block_hadamard(w, block_size=128, dim=-1)
        rot_out = torch.matmul(x_rot, w_rot.t())

        diff = torch.norm(orig_out - rot_out).item() / torch.norm(orig_out).item()
        self.assertLess(diff, 1e-4, f"Rotated output differs from original output: {diff}")

if __name__ == "__main__":
    unittest.main()
