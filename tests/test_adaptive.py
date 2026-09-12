import unittest

import torch

from fquant.adaptive import (
    detect_bifurcation_levels,
    rademacher_signs,
    weighted_groupwise_int4,
    weighted_groupwise_int8,
    weighted_randomized_svd,
    rotate_input_activation,
    rotate_input_weight,
)


class TestAdaptive(unittest.TestCase):
    def test_rademacher_rotation_is_deterministic(self):
        first = rademacher_signs(64, seed=7)
        second = rademacher_signs(64, seed=7)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all((first == -1) | (first == 1)))

    def test_weighted_groupwise_quantization_returns_finite_outputs(self):
        weight = torch.randn(16, 64)
        h = torch.linspace(0.25, 2.0, 64)
        q, scales, reconstruction, metrics = weighted_groupwise_int4(
            weight, h, group_size=64
        )
        self.assertEqual(q.shape, weight.shape)
        self.assertEqual(scales.shape, (16, 1))
        self.assertEqual(reconstruction.shape, weight.shape)
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertGreaterEqual(metrics["weighted_relative_error"], 0.0)

    def test_rademacher_hadamard_pair_preserves_linear_map(self):
        x = torch.randn(3, 128)
        weight = torch.randn(16, 128)
        x_rot = rotate_input_activation(x, seed=1729)
        weight_rot = rotate_input_weight(weight, seed=1729)
        original = torch.nn.functional.linear(x, weight)
        rotated = torch.nn.functional.linear(x_rot, weight_rot)
        self.assertLess(float((original - rotated).abs().max()), 1e-4)

    def test_weighted_svd_reduces_weighted_residual(self):
        left = torch.randn(32, 4)
        right = torch.randn(4, 64)
        residual = left @ right + 0.05 * torch.randn(32, 64)
        h = torch.linspace(0.1, 3.0, 64)
        _, _, metrics = weighted_randomized_svd(residual, h, rank=4, seed=11)
        self.assertGreater(metrics["weighted_capture"], 0.85)

    def test_int8_fallback_is_more_accurate_than_int4(self):
        weight = torch.randn(16, 64)
        h = torch.linspace(0.25, 2.0, 64)
        _, _, _, int4 = weighted_groupwise_int4(weight, h, group_size=64)
        _, _, _, int8 = weighted_groupwise_int8(weight, h, group_size=64)
        self.assertLess(int8["weighted_relative_error"], int4["weighted_relative_error"])

    def test_bifurcation_detector_assigns_rank_map(self):
        rows = [
            {"layer_idx": i, "weighted_relative_error": 0.1 + i * 0.002,
             "spectral_spike": 0.05, "activation_peak": 1.0,
             "weight_outlier": 1.0}
            for i in range(8)
        ]
        rows[5]["weighted_relative_error"] = 0.8
        report = detect_bifurcation_levels(rows)
        self.assertEqual(set(report["rank_by_layer"]), {str(i) for i in range(8)})
        self.assertTrue(all(0 <= row["bifurcation_level"] <= 3 for row in report["layers"]))
        self.assertTrue(all(row["rank"] in {16, 24, 32, 48} for row in report["layers"]))


if __name__ == "__main__":
    unittest.main()
