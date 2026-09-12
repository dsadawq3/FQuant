"""Calibration-aware quantization primitives for the FQuant research path.

The module keeps the representation simple enough for the existing custom
loaders while replacing two weak assumptions in the first implementation:

* scales are selected by minimizing an activation-weighted group error;
* residual factors minimize a diagonal-Hessian proxy rather than an
  unweighted Frobenius error;
* layer budgets are selected from measured, robust spectral and activation
  diagnostics instead of fixed layer-number ranges.

The diagonal Hessian proxy is H ~= diag(E[x^2]). For a linear layer with input
activation x and weight error R, the local output objective is

    E ||x R^T||_2^2 = tr(R H R^T).

This is deliberately a calibration approximation. It is not a claim that a
diagonal Hessian is equivalent to GPTQ's full second-order solver.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .hadamard import apply_block_hadamard


DEFAULT_SCALE_FACTORS: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
DEFAULT_RANK_BY_LEVEL: Mapping[int, int] = {0: 16, 1: 24, 2: 32, 3: 48}


def rademacher_signs(
    size: int,
    seed: int = 1729,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a deterministic +/-1 vector used before a Hadamard rotation."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + 1009 * int(size)) % (2**63 - 1))
    bits = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8)
    signs = bits.to(torch.float32).mul_(2.0).sub_(1.0)
    return signs.to(device=device, dtype=dtype)


def rotate_input_weight(
    weight: torch.Tensor,
    *,
    block_size: int = 128,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply the same input-space transform used by the adaptive Mini loader.

    For an input x and weight W, the transformed pair is x' = x D H and
    W' = W D H, where D is diagonal with +/-1 entries and H is orthonormal.
    Since D^2 = I and H H^T = I, the exact full-precision linear map is
    unchanged.
    """

    if weight.ndim != 2 or weight.shape[1] % block_size:
        return weight
    transformed = weight
    if seed is not None:
        transformed = transformed * rademacher_signs(
            weight.shape[1], seed, dtype=weight.dtype, device=weight.device
        ).unsqueeze(0)
    return apply_block_hadamard(transformed, block_size=block_size, dim=-1)


def rotate_input_activation(
    activation: torch.Tensor,
    *,
    block_size: int = 128,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply the activation-side transform paired with ``rotate_input_weight``."""

    if activation.ndim == 0 or activation.shape[-1] % block_size:
        return activation
    transformed = activation
    if seed is not None:
        transformed = transformed * rademacher_signs(
            activation.shape[-1], seed, dtype=activation.dtype, device=activation.device
        )
    return apply_block_hadamard(transformed, block_size=block_size, dim=-1)


def _normalized_activation_second_moment(
    activation_second_moment: Optional[torch.Tensor],
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if activation_second_moment is None:
        return torch.ones(width, dtype=torch.float32, device=device)
    h = activation_second_moment.detach().to(device=device, dtype=torch.float32).flatten()
    if h.numel() != width:
        raise ValueError(f"activation second moment has {h.numel()} entries; expected {width}")
    h = torch.nan_to_num(h, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-12)
    return h / h.mean().clamp_min(1e-12)


def weighted_groupwise_int4(
    weight: torch.Tensor,
    activation_second_moment: Optional[torch.Tensor] = None,
    *,
    group_size: int = 64,
    scale_factors: Sequence[float] = DEFAULT_SCALE_FACTORS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Quantize a matrix using per-group scales selected by weighted MSE.

    Returns ``(q_int8, scales, reconstruction, metrics)``. The packed format
    is intentionally left to the model-specific writer.
    """

    if weight.ndim != 2:
        raise ValueError(f"expected a 2D matrix, received shape={tuple(weight.shape)}")
    rows, width = weight.shape
    if width % group_size:
        raise ValueError(f"width={width} is not divisible by group_size={group_size}")
    if not scale_factors:
        raise ValueError("scale_factors must not be empty")

    w = weight.detach().to(dtype=torch.float32)
    h = _normalized_activation_second_moment(
        activation_second_moment, width, device=w.device
    )
    groups = w.view(rows, width // group_size, group_size)
    h_groups = h.view(1, width // group_size, group_size)
    base_scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0

    best_error = torch.full(
        base_scale.shape[:-1], float("inf"), dtype=torch.float32, device=w.device
    )
    best_q = torch.zeros_like(groups, dtype=torch.int8)
    best_scale = base_scale.squeeze(-1)

    for factor in scale_factors:
        if factor <= 0:
            raise ValueError(f"scale factors must be positive, received {factor}")
        scale = (base_scale * float(factor)).clamp_min(1e-8)
        q = torch.round(groups / scale).clamp(-8, 7).to(torch.int8)
        reconstruction = q.to(torch.float32) * scale
        error = ((groups - reconstruction).square() * h_groups).sum(dim=-1)
        take = error < best_error
        best_q = torch.where(take.unsqueeze(-1), q, best_q)
        best_scale = torch.where(take, scale.squeeze(-1), best_scale)
        best_error = torch.where(take, error, best_error)

    reconstruction = (best_q.to(torch.float32) * best_scale.unsqueeze(-1)).view(rows, width)
    weighted_denominator = (w.square() * h.unsqueeze(0)).sum().clamp_min(1e-12)
    weighted_relative_error = torch.sqrt(best_error.sum() / weighted_denominator)
    unweighted_relative_error = torch.linalg.vector_norm(w - reconstruction) / torch.linalg.vector_norm(w).clamp_min(1e-12)
    metrics = {
        "weighted_relative_error": float(weighted_relative_error.item()),
        "unweighted_relative_error": float(unweighted_relative_error.item()),
        "mean_scale_factor": float((best_scale / base_scale.squeeze(-1)).mean().item()),
        "clipped_group_fraction": float((best_scale < base_scale.squeeze(-1)).float().mean().item()),
    }
    return best_q.view(rows, width), best_scale.to(torch.bfloat16), reconstruction, metrics


def weighted_groupwise_int8(
    weight: torch.Tensor,
    activation_second_moment: Optional[torch.Tensor] = None,
    *,
    group_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Activation-weighted symmetric INT8 fallback for medium-risk matrices."""

    if weight.ndim != 2:
        raise ValueError(f"expected a 2D matrix, received shape={tuple(weight.shape)}")
    rows, width = weight.shape
    if width % group_size:
        raise ValueError(f"width={width} is not divisible by group_size={group_size}")

    w = weight.detach().to(dtype=torch.float32)
    h = _normalized_activation_second_moment(
        activation_second_moment, width, device=w.device
    )
    groups = w.view(rows, width // group_size, group_size)
    base_scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(groups / base_scale).clamp(-128, 127).to(torch.int8)
    reconstruction = (q.to(torch.float32) * base_scale).view(rows, width)
    denominator = (w.square() * h.unsqueeze(0)).sum().clamp_min(1e-12)
    weighted_error = ((w - reconstruction).square() * h.unsqueeze(0)).sum()
    metrics = {
        "weighted_relative_error": float(torch.sqrt(weighted_error / denominator).item()),
        "unweighted_relative_error": float(
            (torch.linalg.vector_norm(w - reconstruction)
             / torch.linalg.vector_norm(w).clamp_min(1e-12)).item()
        ),
    }
    return q.view(rows, width), base_scale.squeeze(-1).to(torch.bfloat16), reconstruction, metrics


def _weighted_residual(
    residual: torch.Tensor,
    activation_second_moment: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    h = _normalized_activation_second_moment(
        activation_second_moment, residual.shape[1], device=residual.device
    )
    sqrt_h = h.sqrt()
    return residual.to(torch.float32) * sqrt_h.unsqueeze(0), sqrt_h


def weighted_randomized_svd(
    residual: torch.Tensor,
    activation_second_moment: Optional[torch.Tensor],
    rank: int,
    *,
    oversample: int = 8,
    niter: int = 2,
    seed: int = 1729,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Approximate a residual under the diagonal-Hessian objective.

    If R_w = R diag(sqrt(H)), randomized SVD finds A B_w ~= R_w and returns
    B = B_w diag(1/sqrt(H)). Therefore A B approximates R while optimizing
    ``tr(R H R^T)``. Factors are stored in BF16 to match the model loaders.
    """

    if residual.ndim != 2:
        raise ValueError(f"expected a 2D residual, received shape={tuple(residual.shape)}")
    rows, width = residual.shape
    effective_rank = min(max(int(rank), 0), rows, width)
    if effective_rank == 0:
        a = torch.zeros((rows, 0), dtype=torch.bfloat16, device=residual.device)
        b = torch.zeros((0, width), dtype=torch.bfloat16, device=residual.device)
        return a, b, {"rank": 0.0, "weighted_capture": 0.0}

    residual_weighted, sqrt_h = _weighted_residual(residual, activation_second_moment)
    q = min(effective_rank + max(int(oversample), 0), rows, width)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) % (2**63 - 1))
        if q >= min(rows, width):
            u, s, vh = torch.linalg.svd(residual_weighted, full_matrices=False)
            u, s, vh = u[:, :effective_rank], s[:effective_rank], vh[:effective_rank, :]
        else:
            u, s, v = torch.svd_lowrank(residual_weighted, q=q, niter=niter)
            u, s, v = u[:, :effective_rank], s[:effective_rank], v[:, :effective_rank]
            vh = v.transpose(0, 1)

    sqrt_s = s.clamp_min(1e-8).sqrt()
    a = u * sqrt_s.unsqueeze(0)
    b_weighted = sqrt_s.unsqueeze(1) * vh
    b = b_weighted / sqrt_h.clamp_min(1e-6).unsqueeze(0)
    reconstructed = a @ b
    weighted_error = torch.linalg.vector_norm((residual - reconstructed) * sqrt_h.unsqueeze(0))
    weighted_norm = torch.linalg.vector_norm(residual_weighted).clamp_min(1e-12)
    metrics = {
        "rank": float(effective_rank),
        "weighted_capture": float((1.0 - weighted_error / weighted_norm).clamp(min=-1.0, max=1.0).item()),
    }
    return a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous(), metrics


def residual_spectral_spike(
    residual: torch.Tensor,
    activation_second_moment: Optional[torch.Tensor] = None,
    *,
    iterations: int = 3,
    seed: int = 1729,
) -> float:
    """Estimate sigma_1(R_w)^2 / ||R_w||_F^2 with power iteration."""

    if residual.ndim != 2:
        raise ValueError("residual_spectral_spike expects a matrix")
    residual_weighted, _ = _weighted_residual(residual, activation_second_moment)
    denominator = residual_weighted.square().sum().clamp_min(1e-12)
    if denominator.item() == 1e-12 and not bool(residual_weighted.abs().any()):
        return 0.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) % (2**63 - 1))
        vector = torch.randn(residual_weighted.shape[1], device=residual_weighted.device)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
    for _ in range(max(int(iterations), 1)):
        left = residual_weighted @ vector
        left = left / torch.linalg.vector_norm(left).clamp_min(1e-12)
        vector = residual_weighted.transpose(0, 1) @ left
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
    sigma = torch.linalg.vector_norm(residual_weighted @ vector)
    return float((sigma.square() / denominator).clamp(0.0, 1.0).item())


def _robust_z(values: torch.Tensor) -> torch.Tensor:
    median = values.median()
    mad = (values - median).abs().median()
    scale = (1.4826 * mad).clamp_min(1e-6)
    return (values - median) / scale


def detect_bifurcation_levels(
    layer_metrics: Iterable[Mapping[str, float]],
    *,
    rank_by_level: Mapping[int, int] = DEFAULT_RANK_BY_LEVEL,
) -> Dict[str, object]:
    """Assign L0-L3 budgets using robust statistics and change signals.

    The detector combines robust feature z-scores with first-difference and
    second-difference signals. The latter are useful for transitions where a
    layer is important because it sits at a sharp change in the stack, even if
    its absolute score is not the global maximum.
    """

    rows = sorted((dict(item) for item in layer_metrics), key=lambda item: int(item["layer_idx"]))
    if not rows:
        return {"layers": [], "rank_by_layer": {}, "level_counts": {}}

    feature_names = (
        "weighted_relative_error",
        "spectral_spike",
        "activation_peak",
        "weight_outlier",
    )
    feature_z = []
    for feature in feature_names:
        values = torch.tensor([float(row.get(feature, 0.0)) for row in rows], dtype=torch.float32)
        feature_z.append(_robust_z(values))
    base_score = torch.stack(feature_z, dim=1).mean(dim=1)
    if len(rows) == 1:
        jumps = torch.zeros(1)
        curvature = torch.zeros(1)
    else:
        jumps = torch.zeros(len(rows))
        jumps[1:] = (base_score[1:] - base_score[:-1]).abs()
        jumps[:-1] = torch.maximum(jumps[:-1], (base_score[1:] - base_score[:-1]).abs())
        curvature = torch.zeros(len(rows))
        if len(rows) > 2:
            curvature[1:-1] = (base_score[2:] - 2 * base_score[1:-1] + base_score[:-2]).abs()
    jump_z = _robust_z(jumps) if len(rows) > 1 else torch.zeros(1)
    curvature_z = _robust_z(curvature) if len(rows) > 2 else torch.zeros(len(rows))

    quantiles = torch.quantile(base_score, torch.tensor([0.50, 0.75, 0.90]))
    output_layers = []
    counts = {int(level): 0 for level in sorted(rank_by_level)}
    for idx, row in enumerate(rows):
        score = base_score[idx]
        if score >= quantiles[2]:
            level = 3
        elif score >= quantiles[1]:
            level = 2
        elif score >= quantiles[0]:
            level = 1
        else:
            level = 0
        if jump_z[idx] >= 1.5 or curvature_z[idx] >= 1.5:
            level = min(3, level + 1)
        level = min(level, max(rank_by_level))
        counts[level] = counts.get(level, 0) + 1
        output_layers.append({
            **row,
            "risk_score": float(score.item()),
            "jump_score": float(jump_z[idx].item()),
            "curvature_score": float(curvature_z[idx].item()),
            "bifurcation_level": int(level),
            "rank": int(rank_by_level[level]),
        })

    return {
        "layers": output_layers,
        "rank_by_layer": {str(row["layer_idx"]): int(row["rank"]) for row in output_layers},
        "level_counts": counts,
        "thresholds": [float(value.item()) for value in quantiles],
    }
