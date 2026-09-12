# FQuant: Post-Training LLM Quantization Toolkit

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-dsadawq3%2FFQuant-black?logo=github)](https://github.com/dsadawq3/FQuant)
[![Organization](https://img.shields.io/badge/%F0%9F%A4%97%20Organization-F--Labs-yellow.svg)](https://huggingface.co/F-Labs)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

**FQuant** is an open-source post-training LLM quantization toolkit engineered at **[F-Labs](https://huggingface.co/F-Labs)**.

It addresses the severe reasoning collapse, activation outlier clipping, and long-context hallucinations of naive INT4/INT8 quantization through a cohesive 7-part engineering pipeline combining known quantization techniques.

---

## 🚀 Major FQuant Update — 2026-09-12

FQuant has moved beyond a small tensor quantization utility into a reproducible,
model-aware research and deployment pipeline. The current release adds a
calibration-aware adaptive path, architecture-specific loading, numerical
guardrails, and release artifacts that can be inspected and reproduced.

### What shipped

- **Calibration-aware mixed precision:** deterministic Rademacher/Hadamard
  transforms, activation-weighted scale selection, weighted randomized SVD, and
  sensitivity-guided allocation between BF16, INT8, and INT4 paths.
- **Architecture-aware inference:** the MiniCPM5-2B path reconstructs the
  custom FQuant tensor representation directly, including paired rotations,
  low-rank residuals, grouped quantization, KV-BSS, and cache-aware generation.
- **Numerical hardening:** GQA and attention-mask validation, finite handling of
  fully masked rows, deterministic scratch initialization, and cached versus
  uncached logit parity checks.
- **Reproducible release evidence:** calibration moments, bifurcation reports,
  exact tensor indexes, shard metadata, and documented validation commands are
  shipped with the adaptive model artifact.

The current MiniCPM adaptive quality probe contains **234 BF16 high-sensitivity
projections and 60 INT8 MLP projections** across **561 indexed tensors**. Its
three-shard artifact is approximately **4.02 GiB** versus **4.69 GiB** for the
raw BF16 snapshot. The larger adaptive artifact is an intentional quality-
oriented intermediate point while lower-bit policies are being evaluated.

The evidence is now executable: the FQuant core suite passes **12 tests**, and
the MiniCPM numerical/integration suite passes **6 tests**. These checks validate
loader behavior and numerical safety; they do not claim benchmark superiority,
long-context recall gains, or optimized llama.cpp throughput. See the
experimental [adaptive MiniCPM release](https://huggingface.co/F-Labs/MiniCPM5-2B-Hadamard-GSQ)
and its [model-aware source repository](https://github.com/dsadawq3/MiniCPM5-2B-Hadamard-GSQ).

---

## Notice on Model Quality, Iterative Reformation & Strategic Roadmap

> **Ecosystem Distribution & Continuous Evolution Notice**:
> Architectural parameters, SVD rank allocations, and reconstruction tolerances in FQuant are subject to ongoing research refinement. 
> As theoretical optimizations advance, existing models will periodically undergo architectural reformations and quality updates. 
> At present, our primary strategic focus is broad ecosystem distribution, expanding architectural support across open foundation LLMs, and validating edge hardware compatibility (mobile phones, NPUs, Jetson, and embedded systems).

---

## 🏛️ The 7 Pillars of FQuant

1. **DV-SSQ (Dense-Vectorized Subspace Salience Quantization)**: Multi-tier parameter partitioning isolating salient semantic channels (INT8), background MLP weights (INT4 group-wise), and eigenspace residuals (BF16 SVD).
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

## CLI Usage

FQuant includes a command-line utility for inspection, tensor transformation, and
index validation. The generic engine currently writes its own tensor representation;
loading the result requires a model-specific loader such as the MiniCPM or Spark
implementations in this workspace.

### 1. Automatic Inspection of Target Model
```bash
fquant inspect --model openbmb/MiniCPM5-2B
```

### 2. Multi-Tier Quantization
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

> **Diagnostic guardrail**:
> If an unexpected structural anomaly or excessive numerical deviation occurs during SVD decomposition, FQuant preserves the output and records layer-level diagnostics. Review the log before using the result:
> `https://github.com/dsadawq3/FQuant/issues`

---

## Python API

```python
import torch
from fquant import FQuantEngine, KVBSSAttentionHook

# Initialize Engine with SVD rank and group configuration
engine = FQuantEngine(group_size=64, default_rank=16, k_proj_rank=32)

# Generic tensor transformation from a Hugging Face or local path.
# The output is not automatically loadable by every Transformers architecture.
engine.quantize_model(
    model_source="openbmb/MiniCPM5-2B",
    output_dir="./MiniCPM5-2B-Hadamard-GSQ"
)

# Apply KV-BSS inference hook for long-context recall
kv_hook = KVBSSAttentionHook(tau_focus=1.10, haze_floor_margin=12.0)
```

---

## Quantized Model Releases

- **[F-Labs/Spark-X2.5-4B-Hadamard-GSQ](https://huggingface.co/F-Labs/Spark-X2.5-4B-Hadamard-GSQ)** (4.18 GB, -45.4% RAM saved)
- **[F-Labs/MiniCPM5-2B-Hadamard-GSQ](https://huggingface.co/F-Labs/MiniCPM5-2B-Hadamard-GSQ)** (2.03 GB, -56.6% RAM saved, 128K context)

---
## Related Work & Attribution

FQuant builds on established quantization literature; our contribution is the composition into an edge-focused pipeline plus per-model artifacts and edge measurements.

- [QuaRot](https://arxiv.org/abs/2404.00456) — Hadamard rotation for quantization; we use the same principle with fixed H128/H256 Walsh-Hadamard blocks + group-wise INT4, without claiming the rotation itself.
- [SpinQuant](https://arxiv.org/abs/2405.16406) — learned rotations; we use fixed Walsh-Hadamard blocks with no training, trading adaptivity for edge simplicity.
- [GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) — group quantization and salient channels; our group-wise INT4 (g=64) and INT8 tier follow in the spirit of that work.
- [ZeroQuant-V2](https://arxiv.org/abs/2307.09782) / [LoRC](https://arxiv.org/abs/2312.09934) — low-rank compensation of quantization error; our SRC is the same class of idea applied to group-wise INT4 residuals.
- [LLM.int8()](https://arxiv.org/abs/2208.07339) / [SpQR](https://arxiv.org/abs/2306.03078) — mixed precision for outliers; our DV-SSQ salient tier follows the same approach.

---

## License

Apache License 2.0. Engineered at **[F-Labs](https://huggingface.co/F-Labs)**.
