from __future__ import annotations

import numpy as np


def _add_one_two_sided_p(samples: np.ndarray) -> float:
    lower = int(np.sum(samples <= 0.0))
    upper = int(np.sum(samples >= 0.0))
    return float(min(1.0, 2.0 * (1 + min(lower, upper)) / (len(samples) + 1)))


def moving_block_bootstrap_summary(
    values,
    repeats: int = 20_000,
    block_length: int = 20,
    random_state: int = 1,
) -> dict:
    observations = np.asarray(values, dtype=np.float64)
    if (
        observations.ndim != 1
        or len(observations) == 0
        or np.any(~np.isfinite(observations))
    ):
        raise ValueError("values must be a non-empty finite one-dimensional array")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    if int(block_length) <= 0:
        raise ValueError("block_length must be positive")

    sample_count = len(observations)
    selected_block_length = int(block_length)
    block_count = int(np.ceil(sample_count / selected_block_length))
    offsets = np.arange(selected_block_length, dtype=int)
    rng = np.random.default_rng(random_state)
    samples = np.empty(int(repeats), dtype=np.float64)

    for repeat in range(int(repeats)):
        starts = rng.integers(0, sample_count, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % sample_count).reshape(-1)
        samples[repeat] = float(np.mean(observations[indices[:sample_count]]))

    return {
        "estimate": float(np.mean(observations)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_value": _add_one_two_sided_p(samples),
        "observations": int(sample_count),
        "block_length": selected_block_length,
    }
