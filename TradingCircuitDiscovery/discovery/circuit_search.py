from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CandidateNeuron:
    layer: str
    index: int
    score: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.layer}.neuron{self.index}"


@dataclass
class CircuitSearchResult:
    neurons: list[CandidateNeuron]
    effect: float


def greedy_circuit_search(
    candidates: list[CandidateNeuron],
    effect_fn: Callable[[list[CandidateNeuron]], float],
    max_size: int = 5,
    min_relative_gain: float = 0.0,
) -> CircuitSearchResult:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if min_relative_gain < 0:
        raise ValueError("min_relative_gain must be non-negative")
    selected: list[CandidateNeuron] = []
    remaining = list(candidates)
    best_effect = 0.0
    while remaining and len(selected) < max_size:
        scored = []
        for candidate in remaining:
            trial = selected + [candidate]
            scored.append((float(effect_fn(trial)), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        next_effect, next_candidate = scored[0]
        if selected:
            marginal_gain = next_effect - best_effect
            required_gain = abs(best_effect) * float(min_relative_gain)
            if marginal_gain < required_gain:
                break
        selected.append(next_candidate)
        remaining.remove(next_candidate)
        best_effect = next_effect
    return CircuitSearchResult(neurons=selected, effect=best_effect)
