from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankedNeuron:
    index: int
    score: float
    mean_a: float
    mean_b: float


def rank_neurons_by_activation_difference(activations, labels, top_k: int = 20) -> list[RankedNeuron]:
    x = np.asarray(activations, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("activation difference ranking expects two classes")
    mean_a = x[y == classes[0]].mean(axis=0)
    mean_b = x[y == classes[1]].mean(axis=0)
    scores = np.abs(mean_a - mean_b)
    order = np.argsort(-scores)[:top_k]
    return [
        RankedNeuron(index=int(i), score=float(scores[i]), mean_a=float(mean_a[i]), mean_b=float(mean_b[i]))
        for i in order
    ]
