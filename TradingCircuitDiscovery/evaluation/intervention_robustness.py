from __future__ import annotations

import numpy as np


def select_minimal_near_max(candidates: list[dict], fraction: float = 0.95) -> dict:
    if not candidates:
        raise ValueError("At least one circuit candidate is required")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    effects = np.asarray([item["validation_effect"] for item in candidates], dtype=np.float64)
    if np.any(~np.isfinite(effects)):
        raise ValueError("validation effects must be finite")
    threshold = float(effects.max()) * float(fraction)
    eligible = [item for item in candidates if float(item["validation_effect"]) >= threshold]
    return min(eligible, key=lambda item: (len(item["indices"]), -float(item["validation_effect"])))


def build_intervention_values(
    mode: str,
    discovery_activations,
    discovery_labels,
    evaluation_labels,
    indices,
    random_state: int = 1,
) -> np.ndarray:
    discovery = np.asarray(discovery_activations, dtype=np.float64)
    labels = np.asarray(discovery_labels, dtype=int)
    evaluation = np.asarray(evaluation_labels, dtype=int)
    selected = np.asarray(indices, dtype=int)
    if discovery.ndim != 2 or labels.shape != (len(discovery),):
        raise ValueError("discovery activations and labels must be paired")
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    width_values = discovery[:, selected]
    if mode == "discovery_mean":
        values = np.repeat(width_values.mean(axis=0, keepdims=True), len(evaluation), axis=0)
    elif mode == "zero":
        values = np.zeros((len(evaluation), len(selected)), dtype=np.float64)
    elif mode == "opposite_class":
        unique = np.unique(labels)
        if set(unique.tolist()) != {0, 1}:
            raise ValueError("opposite_class intervention requires binary discovery labels")
        class_means = {label: width_values[labels == label].mean(axis=0) for label in (0, 1)}
        values = np.vstack([class_means[1 - int(label)] for label in evaluation])
    elif mode == "matched_noise":
        rng = np.random.default_rng(random_state)
        mean = width_values.mean(axis=0)
        std = np.maximum(width_values.std(axis=0), 1e-8)
        values = rng.normal(loc=mean, scale=std, size=(len(evaluation), len(selected)))
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")
    return values.astype(np.float32)


def standardized_intervention_distance(values, discovery_activations, indices) -> dict[str, float]:
    replacements = np.asarray(values, dtype=np.float64)
    discovery = np.asarray(discovery_activations, dtype=np.float64)
    selected = np.asarray(indices, dtype=int)
    if replacements.ndim == 1:
        replacements = replacements.reshape(1, -1)
    reference = discovery[:, selected]
    mean = reference.mean(axis=0)
    std = np.maximum(reference.std(axis=0), 1e-8)
    per_row = np.sqrt(np.mean(((replacements - mean) / std) ** 2, axis=1))
    return {
        "mean_standardized_distance": float(per_row.mean()),
        "max_standardized_distance": float(per_row.max()),
        "fraction_above_three_sigma": float(np.mean(per_row > 3.0)),
    }

