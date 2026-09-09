# ⚡ FQuant: High-Precision Post-Training LLM Quantization Framework

[![F-Labs Organization](https://img.shields.io/badge/%F0%9F%A4%97%20Organization-F--Labs-yellow.svg)](https://huggingface.co/F-Labs)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-dsadawq3%2FFQuant-black?logo=github)](https://github.com/dsadawq3/FQuant)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

**FQuant** is an open-source post-training LLM quantization and attention-sharpening framework engineered at **F-Labs**. 

It eliminates the severe reasoning collapse, activation outlier clipping, and long-context hallucinations of naive INT4/INT8 quantization through a cohesive 7-pillar mathematical architecture.

---

## 🏛️ The 7 Pillars of FQuant

1. **DV-SSQ (Dense-Vectorized Subspace Salience Quantization)**: Multi-tier parameter partitioning isolating salient channels (INT8), background MLP weights (INT4 GSQ), and eigenspace residuals (BF16 SVD).
2. **KV-BSS (Key-Value Binding Softmax Sharpening)**: Focus temperature scaling ($\tau_{\text{focus}} = 1.10$) and Attention Haze floor suppression ($< \max - 12.0$) to prevent entity and structured key-value hallucinations (`["key"] => "value"`) on contexts up to 128K tokens.
3. **Zero-Compression Shield**: 100% pure BF16 isolation for all RMSNorms, biases, and token embeddings.
4. **Key-Projection Exponential Sensitivity Defense**: Doubled SVD rank ($r=32$) on `k_proj` to prevent exponential amplification of quantization noise in $\exp(Q K^T / \sqrt{d})$.
5. **Spectral Entropy Bifurcation Detection**: Automatic singular-value entropy calculation allocating dynamic rank budgets to deep abstraction reasoning circuits.
6. **Orthonormal Walsh-Hadamard Spin Rotation**: Coordinate rotation ($W' = W H^T, X' = X H$) eliminating coordinate-aligned activation outliers by over 80%.
7. **Null-Space Noise Confinement**: Directing residual quantization error into the model's non-semantic null space $\ker(X)$.

---

## 📦 Installation

```bash
git clone https://github.com/dsadawq3/FQuant.git
cd FQuant
pip install -e .
```

---

## 🚀 Quick Usage

```python
import torch
from fquant import FQuantEngine, KVBSSAttentionHook

# 1. Initialize Engine
engine = FQuantEngine(group_size=64, default_rank=16, k_proj_rank=32)

# 2. Quantize any linear projection weight
weight = torch.randn(2048, 2048, dtype=torch.bfloat16)
result = engine.quantize_parameter("model.layers.0.self_attn.q_proj.weight", weight, layer_idx=0, total_layers=32)

print("Packed INT4 uint8 shape:", result["qweight_packed"].shape)
print("SVD Factor A shape:", result["svd_a"].shape)
print("SVD Factor B shape:", result["svd_b"].shape)

# 3. Apply KV-BSS Attention Hook in long-context inference
kv_hook = KVBSSAttentionHook(tau_focus=1.10, haze_floor_margin=12.0)
query = torch.randn(1, 16, 128, 128)
key = torch.randn(1, 2, 128, 128)
value = torch.randn(1, 2, 128, 128)

context = kv_hook(query, key, value)
print("Context output shape:", context.shape)
```

---

## 📄 License

Apache License 2.0. Engineered at **F-Labs**.
