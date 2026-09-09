"""
Spectral Entropy Bifurcation Detector.
Computes singular value distribution entropy to detect deep abstraction circuits and allocate dynamic eigenspace budget.
"""

import math
import torch

class SpectralEntropyBifurcationDetector:
    @staticmethod
    def calculate_spectral_entropy(weight: torch.Tensor) -> float:
        try:
            s = torch.linalg.svdvals(weight.float())
            p = s / s.sum().clamp(min=1e-7)
            entropy = -(p * torch.log(p.clamp(min=1e-12))).sum().item()
            return entropy
        except Exception:
            return 0.0

    @staticmethod
    def get_layer_rank(layer_idx: int, total_layers: int, is_k_proj: bool = False, default_rank: int = 16) -> int:
        if is_k_proj:
            return 32  # Exponential sensitivity defense
        # Bifurcation hub: mid-deep reasoning circuits (layer 35% to 70%)
        start_bifurcation = int(total_layers * 0.33)
        end_bifurcation = int(total_layers * 0.67)
        if start_bifurcation <= layer_idx <= end_bifurcation:
            return default_rank + 8  # Expanded rank 24
        return default_rank
