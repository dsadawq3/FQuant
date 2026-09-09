import unittest
import torch
from fquant.kv_bss import KVBSSAttentionHook

class TestKVBSS(unittest.TestCase):
    def test_attention_hook_forward(self):
        hook = KVBSSAttentionHook(tau_focus=1.10, haze_floor_margin=12.0)
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 2, 16, 64)  # GQA 2:1
        v = torch.randn(2, 2, 16, 64)

        out = hook(q, k, v)
        self.assertEqual(out.shape, (2, 4, 16, 64))
        self.assertFalse(torch.isnan(out).any())

if __name__ == "__main__":
    unittest.main()
