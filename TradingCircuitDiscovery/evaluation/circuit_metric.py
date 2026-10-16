from __future__ import annotations

import numpy as np


def circuit_causal_score(original_probs, patched_probs, action_index: int) -> float:
    original = np.asarray(original_probs, dtype=np.float64)
    patched = np.asarray(patched_probs, dtype=np.float64)
    return float(abs(original[action_index] - patched[action_index]))


def circuit_sparsity(neurons) -> int:
    return int(len(list(neurons)))


def circuit_stability(circuit_a, circuit_b) -> float:
    a = set(circuit_a)
    b = set(circuit_b)
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b))
