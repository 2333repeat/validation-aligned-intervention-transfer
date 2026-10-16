from __future__ import annotations

import torch


def ablate_neurons(activation: torch.Tensor, indices: list[int], baseline: float | torch.Tensor = 0.0) -> torch.Tensor:
    result = activation.clone()
    result[..., indices] = baseline
    return result
