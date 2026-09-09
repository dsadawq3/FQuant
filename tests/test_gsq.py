import unittest
import torch
from fquant.gsq import quantize_gsq_int4, pack_int4_to_uint8, unpack_uint8_to_int4

class TestGSQ(unittest.TestCase):
    def test_packing_roundtrip(self):
        # Create random int8 values in [-8, 7]
        orig_int8 = torch.randint(-8, 8, (64, 128), dtype=torch.int8)
        packed = pack_int4_to_uint8(orig_int8)
        self.assertEqual(packed.shape, (64, 64))
        self.assertEqual(packed.dtype, torch.uint8)

        unpacked = unpack_uint8_to_int4(packed)
        self.assertEqual(unpacked.shape, (64, 128))
        self.assertEqual(unpacked.dtype, torch.int8)

        diff = torch.max(torch.abs(orig_int8 - unpacked)).item()
        self.assertEqual(diff, 0, "Unpacked INT4 values do not match original bit-for-bit")

    def test_quantize_scale(self):
        w = torch.randn(128, 256)
        q_w, scale, w_dequant = quantize_gsq_int4(w, group_size=64)
        self.assertTrue(torch.all(q_w >= -8))
        self.assertTrue(torch.all(q_w <= 7))
        self.assertEqual(scale.shape, (128, 4))

if __name__ == "__main__":
    unittest.main()
