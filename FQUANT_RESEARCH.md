# FQuant adaptive research track

This document is the handoff contract for the calibration-aware FQuant path.
It separates published methods from the hypotheses added by this repository so
that another model can reproduce the work without treating a promising metric
as a proven quality claim.

## Problem definition

The first MiniCPM release used a fixed rank rule: layers 14 through 27 received
rank 24 and the remaining linear layers received rank 16. That rule is simple,
but it does not measure whether a particular layer actually carries more
quantization risk. The adaptive track measures each layer after the same input
rotation used by the runtime loader, then assigns one of four budgets:

| Level | Default residual rank | Meaning in this experiment |
| ---: | ---: | --- |
| L0 | 16 | low measured residual risk |
| L1 | 24 | ordinary risk |
| L2 | 32 | elevated risk |
| L3 | 48 | high risk or a sharp transition |

The rank detector is a heuristic diagnostic. “Bifurcation” describes a detected
change in a measured risk curve; it does not assert a phase transition in the
neural network.

## Exact transform and objective

For a linear layer `y = x W^T`, the release uses a deterministic diagonal
Rademacher matrix `D`, whose diagonal entries are independently selected from
`{-1,+1}`, followed by a block orthonormal Walsh-Hadamard matrix `H`:

```text
x' = x D H
W' = W D H
```

Because `D^2 = I` and `H H^T = I`, the full precision map is preserved:

```text
x' W'^T = x D H H^T D W^T = x W^T
```

The implementation uses block size 128 and seed 1729 by default. The seed is
stored in `config.json`, and the loader regenerates the signs instead of
serializing a redundant vector.

Let `R = W' - W_hat` be the groupwise INT4 error. For calibration activations,
the local squared output error is approximated with a diagonal second moment:

```text
H_x ~= diag(E[x'^T x'])
J(R) = tr(R H_x R^T)
     = || R diag(sqrt(h)) ||_F^2
```

This is a diagonal Hessian proxy. It is useful because it is cheap and gives a
clear weighting for scales and residual factors, but it is not GPTQ's full
second-order block solver.

For each row and group, candidate scale multipliers
`{0.50, 0.60, 0.70, 0.80, 0.90, 1.00}` are evaluated and the candidate with
the smallest activation-weighted squared error is selected. The residual
factorization applies randomized SVD to `R diag(sqrt(h))` and unweights the
right factor:

```text
R diag(sqrt(h)) ~= A B_w
B = B_w diag(1 / sqrt(h))
```

The runtime computes `x' W_hat^T + (x' B^T) A^T`.

## Risk and transition detector

For every quantized linear tensor the calibration pass records:

* weighted relative INT4 error;
* unweighted relative error;
* residual spectral spike, estimated as `sigma_1(R_w)^2 / ||R_w||_F^2`;
* activation peak, `max(h) / mean(h)`;
* weight outlier ratio, `max(abs(W)) / RMS(W)`.

Layer features are aggregated by the mean across that layer's projections.
Each feature receives a robust z-score using median and MAD. The base risk is
the mean feature z-score. Absolute first differences and second finite
differences are added as transition signals. Quantiles at 50%, 75%, and 90%
produce L1, L2, and L3; a robust jump or curvature z-score of at least 1.5
promotes a layer by one level, capped at L3.

This choice is intentionally inspectable. It can be replaced by a change-point
model, Bayesian state-space detector, or a learned budget policy without
changing the serialized loader format.

## What is grounded in prior work

The following are facts about the cited methods, followed by the local design
choice in this repository:

* [GPTQ](https://arxiv.org/abs/2210.17323) formulates accurate post-training
  weight quantization using approximate second-order information. FQuant uses
  a much cheaper diagonal moment proxy rather than claiming to implement GPTQ.
* [AWQ](https://arxiv.org/abs/2306.00978) motivates activation-aware protection
  of salient weight channels. FQuant uses activation moments to choose among
  scale candidates and to weight residual fitting.
* [QuaRot](https://arxiv.org/abs/2404.00456) studies orthogonal rotations for
  outlier-free low-bit LLM inference. FQuant uses the same orthogonal-rotation
  principle with a fixed block Hadamard transform plus a deterministic sign
  diagonal.
* [SpinQuant](https://arxiv.org/abs/2405.16406) studies learned rotations.
  FQuant's signs are generated without training, so this track does not claim
  SpinQuant-level rotation optimization.
* [OmniQuant](https://arxiv.org/abs/2308.13137) studies calibration of
  quantization parameters. The adaptive script follows that calibration spirit
  while keeping the model-specific release training-free.
* [SpQR](https://arxiv.org/abs/2306.03078) demonstrates mixed-precision
  protection for sensitive weights. FQuant's BF16 shield for norms and
  embeddings is a separate, simpler mixed-precision policy.
* [QuIP#](https://arxiv.org/abs/2402.04396) studies incoherence and low-bit
  quantization. The FQuant rotation and residual diagnostics are compatible
  with that motivation, but the implementation and quality claims remain local
  to this repository.

## Reproduction protocol

From the MiniCPM repository:

```powershell
python quantize_minicpm_adaptive.py `
  --raw_dir ..\..\models\MiniCPM5-2B-raw `
  --out_dir ..\..\models\MiniCPM5-2B-Hadamard-GSQ-adaptive
```

The output includes the custom loader, `config.json`, the model index and
shards, `adaptive_bifurcation_report.json`, and the small
`adaptive_activation_second_moments.pt` calibration artifact. The old
`MiniCPM5-2B-Hadamard-GSQ` directory is not modified.

Validation must use the same tokenized inputs for every variant and report:

1. index-to-shard key coverage;
2. BF16 forward finiteness and logits shape;
3. logit MSE, relative error, cosine similarity, top-1 agreement, and KL;
4. greedy generation with a fixed prompt and fixed token budget;
5. peak resident memory and serialized bytes;
6. a task metric or benchmark with a stated dataset and split.

The adaptive track is successful only if it improves a held-out quality metric
or memory/latency tradeoff at comparable size. Calibration reconstruction
error alone is an engineering signal, not evidence of better language quality.

## Open hypotheses and ablations

The next model should run these ablations before changing the default release:

* fixed Hadamard versus Rademacher plus Hadamard, with the same seed protocol;
* unweighted scales versus weighted scales;
* unweighted SVD versus diagonal-Hessian-weighted SVD;
* fixed ranks versus detected L0-L3 ranks;
* diagonal moments versus block-diagonal or low-rank curvature estimates;
* `niter` 1, 2, and 4 for randomized SVD;
* calibration mixtures emphasizing English, Chinese, Russian, code, and long
  context separately;
* rank/size Pareto curves for L0-L3 budgets;
* BF16 shield variants for embeddings, norms, and KV-sensitive projections.

The main technical risk is that a diagonal moment can overweight channels that
are frequent in calibration while missing cross-channel curvature. The main
systems risk is CPU calibration and SVD cost. Both risks must be measured and
reported with the release artifact.

The current MiniCPM experiment also has a mixed-precision policy: all attention
projections and all MLP down projections are dense BF16 after the reversible
rotation; L2/L3 MLP gate/up projections are dense BF16; remaining gate/up
projections use groupwise INT8. Only the latter path is retained as a smaller
INT4 candidate in the code, so the resulting 3.67 GiB artifact is a quality
probe rather than the final compression target. A later Pareto sweep can move
those sets back toward INT4 once held-out quality is measured.
