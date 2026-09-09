"""
KV-BSS: Key-Value Binding Softmax Sharpening.
Intervenes at the attention layer to harden the hallucination threshold on structured bindings
and eliminate attention haze in ultra-long contexts (up to 128k tokens).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVBSSAttentionHook(nn.Module):
    """
    Applies:
    1. Focus Temperature Scaling (tau_focus = 1.10)
    2. Dynamic Attention Haze Suppression (scores < max - haze_floor_margin -> -inf)
    """
    def __init__(self, tau_focus: float = 1.10, haze_floor_margin: float = 12.0):
        super().__init__()
        self.tau_focus = tau_focus
        self.haze_floor_margin = haze_floor_margin

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask=None, scaling=None):
        b, h, q_len, d = query.shape
        b, kv_h, kv_len, _ = key.shape
        
        if h != kv_h:
            num_repeat = h // kv_h
            key = key.repeat_interleave(num_repeat, dim=1)
            value = value.repeat_interleave(num_repeat, dim=1)
            
        if scaling is None:
            scaling = 1.0 / math.sqrt(d)
            
        scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
        scores = scores * self.tau_focus
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        max_scores = torch.amax(scores, dim=-1, keepdim=True)
        haze_mask = scores < (max_scores - self.haze_floor_margin)
        scores = scores.masked_fill(haze_mask, -1e4)
        
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(probs, value)
        return out

def apply_kv_bss_attention(query, key, value, attention_mask=None, tau_focus=1.10, haze_floor_margin=12.0):
    hook = KVBSSAttentionHook(tau_focus=tau_focus, haze_floor_margin=haze_floor_margin)
    return hook(query, key, value, attention_mask=attention_mask)
