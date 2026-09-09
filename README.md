# ⚡ FQuant: High-Precision Post-Training LLM Quantization Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-dsadawq3%2FFQuant-black?logo=github)](https://github.com/dsadawq3/FQuant)
[![Organization](https://img.shields.io/badge/%F0%9F%A4%97%20Organization-F--Labs-yellow.svg)](https://huggingface.co/F-Labs)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

**FQuant** is an open-source post-training LLM quantization and attention-sharpening framework engineered at **[F-Labs](https://huggingface.co/F-Labs)**. 

It eliminates the severe reasoning collapse, activation outlier clipping, and long-context hallucinations of naive INT4/INT8 quantization through a cohesive 7-pillar mathematical architecture.

---

## Notice on Model Quality, Iterative Reformation & Strategic Roadmap

> **Ecosystem Distribution & Continuous Evolution Notice**:
> Architectural parameters, SVD rank allocations, and reconstruction tolerances in FQuant are subject to ongoing research refinement. 
> As theoretical optimizations advance, existing models will periodically undergo architectural reformations and quality updates. 
> At present, our primary strategic focus is broad ecosystem distribution, expanding architectural support across open foundation LLMs, and validating edge hardware compatibility (mobile phones, NPUs, Jetson, and embedded systems).

---

## 🏛️ The 7 Pillars of FQuant

1. **DV-SSQ (Dense-Vectorized Subspace Salience Quantization)**: Multi-tier parameter partitioning isolating salient semantic channels (INT8), background MLP weights (INT4 GSQ), and eigenspace residuals (BF16 SVD).
2. **KV-BSS (Key-Value Binding Softmax Sharpening)**: Focus temperature scaling (τ_focus = 1.10) and Attention Haze floor suppression (< max - 12.0) preventing structured key-value hallucinations (`["key"] => "value"`) on contexts up to 128K tokens.
3. **Zero-Compression Shield**: 100% pure BF16 isolation for all RMSNorms, biases, and token embeddings, eliminating cumulative phase drift across deep transformers.
4. **Key-Projection Exponential Sensitivity Defense**: Doubled SVD rank (`r = 32`) on `k_proj` to neutralize exponential noise amplification in exp(Q · Kᵀ / √d).
5. **Spectral Entropy Bifurcation Detection**: Automatic singular-value entropy calculation allocating dynamic rank budgets to deep abstraction reasoning circuits.
6. **Orthonormal Walsh-Hadamard Spin Rotation**: Coordinate rotation (W' = W · Hᵀ, X' = X · H) eliminating coordinate-aligned activation outliers by over 80%.
7. **Null-Space Noise Confinement**: Directing residual quantization error into the model's non-semantic null space ker(X).

---

## Installation

```bash
git clone https://github.com/dsadawq3/FQuant.git
cd FQuant
pip install -e .
```

---

## 🚀 CLI Usage (Production-Ready Pipeline)

FQuant includes an autonomous command-line utility with self-healing inspection:

### 1. Automatic Inspection of Target Model
```bash
fquant inspect --model openbmb/MiniCPM5-2B
```

### 2. Autonomous Multi-Tier Quantization
```bash
fquant quantize \
    --model openbmb/MiniCPM5-2B \
    --output ./quantized_model \
    --group-size 64 \
    --default-rank 16 \
    --k-proj-rank 32
```

### 3. Verification & Diagnostic Anomaly Detection
```bash
fquant verify --model ./quantized_model
```

> **Built-in Diagnostic Guardrail**:
> If an unexpected structural anomaly or excessive numerical deviation occurs during SVD decomposition, FQuant automatically protects the state, outputs layer-level diagnostics, and prompts the user to file a telemetry issue at:
> `https://github.com/dsadawq3/FQuant/issues`

---

## Python API

```python
import torch
from fquant import FQuantEngine, KVBSSAttentionHook

# Initialize Engine with SVD rank and group configuration
engine = FQuantEngine(group_size=64, default_rank=16, k_proj_rank=32)

# Full model quantization from Hugging Face or local path
engine.quantize_model(
    model_source="openbmb/MiniCPM5-2B",
    output_dir="./MiniCPM5-2B-Hadamard-GSQ"
)

# Apply KV-BSS inference hook for long-context recall
kv_hook = KVBSSAttentionHook(τ_focus=1.10, haze_floor_margin=12.0)
```

---

## Quantized Model Releases

- **[F-Labs/Spark-X2.5-4B-Hadamard-GSQ](https://huggingface.co/F-Labs/Spark-X2.5-4B-Hadamard-GSQ)** (4.18 GB, -45.4% RAM saved)
- **[F-Labs/MiniCPM5-2B-Hadamard-GSQ](https://huggingface.co/F-Labs/MiniCPM5-2B-Hadamard-GSQ)** (2.03 GB, -56.6% RAM saved, 128K context)

---

## License

Apache License 2.0. Engineered at **[F-Labs](https://huggingface.co/F-Labs)**.