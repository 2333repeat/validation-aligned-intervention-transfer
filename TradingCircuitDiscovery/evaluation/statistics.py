from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float


def shuffled_labels(labels, random_state: int = 1) -> np.ndarray:
    values = np.asarray(labels).copy()
    return np.random.default_rng(random_state).permutation(values)


def moving_block_bootstrap_indices(
    length: int,
    block_size: int,
    random_state: int = 1,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rng = np.random.default_rng(random_state)
    block_count = int(np.ceil(length / block_size))
    starts = rng.integers(0, length, size=block_count)
    indices = np.concatenate(
        [(start + np.arange(block_size, dtype=int)) % length for start in starts]
    )
    return indices[:length]


def paired_bootstrap_ci(
    targeted,
    control,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 1,
) -> BootstrapResult:
    targeted_values = np.asarray(targeted, dtype=np.float64)
    control_values = np.asarray(control, dtype=np.float64)
    if targeted_values.shape != control_values.shape or targeted_values.ndim != 1:
        raise ValueError("targeted and control must be paired one-dimensional arrays")
    if len(targeted_values) == 0:
        raise ValueError("paired bootstrap requires at least one observation")
    differences = targeted_values - control_values
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(differences), size=(int(n_bootstrap), len(differences)))
    samples = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        mean_difference=float(differences.mean()),
        ci_low=float(np.quantile(samples, alpha)),
        ci_high=float(np.quantile(samples, 1.0 - alpha)),
    )


def benjamini_hochberg(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a finite one-dimensional array in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted
