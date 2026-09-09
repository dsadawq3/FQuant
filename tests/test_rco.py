import unittest
import torch
from fquant.rco_svd import compute_svd_residual_compensation

class TestRCO(unittest.TestCase):
    def test_residual_energy_reduction(self):
        residual = torch.randn(128, 256)
        factor_a, factor_b, recon = compute_svd_residual_compensation(residual, rank=16)
        self.assertEqual(factor_a.shape, (128, 16))
        self.assertEqual(factor_b.shape, (16, 256))

        unexplained = residual - recon
        res_norm = torch.norm(residual).item()
        unexplained_norm = torch.norm(unexplained).item()
        self.assertLess(unexplained_norm, res_norm, "SVD residual compensation did not reduce residual norm")

if __name__ == "__main__":
    unittest.main()
