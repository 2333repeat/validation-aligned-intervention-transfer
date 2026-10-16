from __future__ import annotations

import numpy as np

from TradingCircuitDiscovery.evaluation.circuit_metric import circuit_causal_score


def compare_action_effect(original_probs, intervened_probs, action_index: int) -> dict[str, float]:
    return {
        "causal_effect": circuit_causal_score(original_probs, intervened_probs, action_index),
        "original_probability": float(original_probs[action_index]),
        "intervened_probability": float(intervened_probs[action_index]),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _action_margin(scores: np.ndarray, actions: np.ndarray) -> np.ndarray:
    selected = scores[np.arange(len(scores)), actions]
    masked = scores.copy()
    masked[np.arange(len(scores)), actions] = -np.inf
    return selected - masked.max(axis=1)


def _paired_score_arrays(original_scores, intervened_scores) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(original_scores, dtype=np.float64)
    intervened = np.asarray(intervened_scores, dtype=np.float64)
    if original.shape != intervened.shape or original.ndim != 2:
        raise ValueError("score arrays must be paired two-dimensional arrays")
    if original.shape[1] < 2:
        raise ValueError("score arrays must contain at least two actions")
    return original, intervened


def _normalized_probabilities(scores: np.ndarray, probability_scores: bool) -> np.ndarray:
    probabilities = scores.copy() if probability_scores else _softmax(scores)
    probabilities = np.maximum(probabilities, 0.0)
    return probabilities / np.maximum(
        probabilities.sum(axis=1, keepdims=True), 1e-12
    )


def _row_jsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    kl_left = np.sum(
        left * np.log((left + 1e-12) / (midpoint + 1e-12)), axis=1
    )
    kl_right = np.sum(
        right * np.log((right + 1e-12) / (midpoint + 1e-12)), axis=1
    )
    return 0.5 * (kl_left + kl_right)


def action_score_profile(
    original_scores,
    intervened_scores,
    probability_scores: bool = False,
) -> dict[str, np.ndarray]:
    original, intervened = _paired_score_arrays(original_scores, intervened_scores)
    original_actions = original.argmax(axis=1)
    intervened_actions = intervened.argmax(axis=1)
    original_margin = _action_margin(original, original_actions)
    intervened_margin = _action_margin(intervened, original_actions)
    original_probabilities = _normalized_probabilities(original, probability_scores)
    intervened_probabilities = _normalized_probabilities(intervened, probability_scores)
    action_count = int(original.shape[1])
    transition_counts = np.zeros((action_count, action_count), dtype=int)
    np.add.at(transition_counts, (original_actions, intervened_actions), 1)
    return {
        "original_actions": original_actions,
        "intervened_actions": intervened_actions,
        "signed_margin_change": intervened_margin - original_margin,
        "probability_shift": intervened_probabilities - original_probabilities,
        "per_state_jsd": _row_jsd(original_probabilities, intervened_probabilities),
        "transition_counts": transition_counts,
        "original_probabilities": original_probabilities,
        "intervened_probabilities": intervened_probabilities,
    }


def _finite_float_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _vector_jsd(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(1, -1)
    right = np.asarray(right, dtype=np.float64).reshape(1, -1)
    left /= np.maximum(left.sum(axis=1, keepdims=True), 1e-12)
    right /= np.maximum(right.sum(axis=1, keepdims=True), 1e-12)
    return float(_row_jsd(left, right)[0])


def compare_transfer_profiles(
    original_scores,
    candidate_scores,
    native_scores,
    probability_scores: bool = False,
    epsilon: float = 1e-12,
) -> dict[str, float | list[list[int]] | None]:
    candidate = action_score_profile(
        original_scores,
        candidate_scores,
        probability_scores=probability_scores,
    )
    native = action_score_profile(
        original_scores,
        native_scores,
        probability_scores=probability_scores,
    )
    candidate_margin = candidate["signed_margin_change"]
    native_margin = native["signed_margin_change"]
    if (
        len(candidate_margin) >= 2
        and np.std(candidate_margin) > epsilon
        and np.std(native_margin) > epsilon
    ):
        candidate_centered = candidate_margin - candidate_margin.mean()
        native_centered = native_margin - native_margin.mean()
        correlation = float(
            np.dot(candidate_centered, native_centered)
            / (
                np.linalg.norm(candidate_centered)
                * np.linalg.norm(native_centered)
            )
        )
    else:
        correlation = float("nan")
    sign_mask = np.maximum(np.abs(candidate_margin), np.abs(native_margin)) > epsilon
    sign_agreement = (
        float(np.mean(np.sign(candidate_margin[sign_mask]) == np.sign(native_margin[sign_mask])))
        if np.any(sign_mask)
        else float("nan")
    )
    original_actions = candidate["original_actions"]
    transition_mask = native["intervened_actions"] != original_actions
    transition_agreement = (
        float(
            np.mean(
                candidate["intervened_actions"][transition_mask]
                == native["intervened_actions"][transition_mask]
            )
        )
        if np.any(transition_mask)
        else float("nan")
    )
    transition_jsd = _vector_jsd(
        candidate["transition_counts"], native["transition_counts"]
    )
    distribution_jsd = float(
        np.mean(
            _row_jsd(
                candidate["intervened_probabilities"],
                native["intervened_probabilities"],
            )
        )
    )
    candidate_shift = candidate["probability_shift"].reshape(-1)
    native_shift = native["probability_shift"].reshape(-1)
    denominator = float(np.linalg.norm(candidate_shift) * np.linalg.norm(native_shift))
    shift_cosine = (
        float(np.dot(candidate_shift, native_shift) / denominator)
        if denominator > epsilon
        else float("nan")
    )
    return {
        "signed_margin_correlation_to_native": _finite_float_or_none(correlation),
        "signed_margin_sign_agreement_to_native": _finite_float_or_none(
            sign_agreement
        ),
        "conditional_transition_agreement_to_native": _finite_float_or_none(
            transition_agreement
        ),
        "native_transition_observations": float(np.sum(transition_mask)),
        "transition_jsd_to_native": _finite_float_or_none(transition_jsd),
        "intervened_distribution_jsd_to_native": _finite_float_or_none(
            distribution_jsd
        ),
        "probability_shift_cosine_to_native": _finite_float_or_none(shift_cosine),
        "candidate_transition_counts": candidate["transition_counts"].tolist(),
        "native_transition_counts": native["transition_counts"].tolist(),
    }


def compare_action_score_effect(original_scores, intervened_scores, probability_scores: bool = False) -> dict[str, float]:
    profile = action_score_profile(
        original_scores,
        intervened_scores,
        probability_scores=probability_scores,
    )
    return {
        "action_flip_rate": float(
            np.mean(profile["original_actions"] != profile["intervened_actions"])
        ),
        "mean_absolute_margin_change": float(
            np.mean(np.abs(profile["signed_margin_change"]))
        ),
        "mean_signed_margin_change": float(
            np.mean(profile["signed_margin_change"])
        ),
        "mean_jsd": float(np.mean(profile["per_state_jsd"])),
    }


def action_score_observation_effects(
    original_scores,
    intervened_scores,
    probability_scores: bool = False,
) -> dict[str, list[float]]:
    profile = action_score_profile(
        original_scores,
        intervened_scores,
        probability_scores=probability_scores,
    )
    signed = np.asarray(profile["signed_margin_change"], dtype=float)
    flips = (
        np.asarray(profile["original_actions"])
        != np.asarray(profile["intervened_actions"])
    ).astype(float)
    return {
        "absolute_margin_change": np.abs(signed).tolist(),
        "signed_margin_change": signed.tolist(),
        "action_flip": flips.tolist(),
        "jsd": np.asarray(profile["per_state_jsd"], dtype=float).tolist(),
    }


def matched_random_circuits(
    width: int,
    size: int,
    repeats: int,
    excluded: list[int] | None = None,
    random_state: int = 1,
) -> list[list[int]]:
    excluded_set = set(excluded or [])
    available = np.asarray([index for index in range(width) if index not in excluded_set], dtype=int)
    if size <= 0 or size > len(available):
        raise ValueError("Random circuit size must fit within non-excluded neurons")
    rng = np.random.default_rng(random_state)
    return [sorted(rng.choice(available, size=size, replace=False).tolist()) for _ in range(repeats)]
