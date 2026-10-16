from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import numpy as np
import torch

from TradingCircuitDiscovery.evaluation.representation_stability import (
    _linear_sum_assignment,
)


ALIGNMENT_METHODS = (
    "identity",
    "hungarian_abs",
    "hungarian_signed",
    "procrustes",
    "cca",
    "optimal_transport",
    "weight_matching",
    "ridge",
)


def _layer_order(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", name)
    return (match.group(1), int(match.group(2))) if match else (name, 0)


def checkpoint_weight_signatures(
    checkpoint_path: str | Path,
    layer: str,
) -> np.ndarray:
    if "." not in str(layer):
        raise ValueError("layer must include checkpoint component and module name")
    component, module_name = str(layer).split(".", 1)
    payload = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or component not in payload:
        raise KeyError(f"checkpoint has no component {component!r}")
    state = payload[component]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint component {component!r} is not a state dict")
    incoming_key = f"{module_name}.weight"
    bias_key = f"{module_name}.bias"
    if incoming_key not in state:
        raise KeyError(f"checkpoint has no layer weight {incoming_key!r}")
    incoming = torch.as_tensor(state[incoming_key]).detach().cpu().double()
    if incoming.ndim != 2:
        raise ValueError("selected layer weight must be two-dimensional")
    unit_count = int(incoming.shape[0])
    if bias_key in state:
        bias = torch.as_tensor(state[bias_key]).detach().cpu().double().reshape(-1, 1)
    else:
        bias = torch.zeros((unit_count, 1), dtype=torch.float64)
    if len(bias) != unit_count:
        raise ValueError("selected layer bias does not match its output width")
    selected_order = _layer_order(module_name)
    candidates = []
    for key, value in state.items():
        if not str(key).endswith(".weight") or str(key) == incoming_key:
            continue
        tensor = torch.as_tensor(value).detach().cpu()
        candidate_name = str(key)[: -len(".weight")]
        if tensor.ndim == 2 and int(tensor.shape[1]) == unit_count:
            order = _layer_order(candidate_name)
            if order > selected_order:
                candidates.append((order, tensor.double()))
    outgoing = None
    if candidates:
        outgoing = min(candidates, key=lambda item: item[0])[1].T
    parts = [incoming, bias]
    if outgoing is not None:
        parts.append(outgoing)
    return torch.cat(parts, dim=1).numpy()


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 3:
        raise ValueError("alignment matrices require at least three paired rows")
    centered = array - array.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def _paired_matrices(reference, target) -> tuple[np.ndarray, np.ndarray]:
    left = _standardize(reference)
    right = _standardize(target)
    if len(left) != len(right):
        raise ValueError("alignment matrices must share paired observations")
    return left, right


def _correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(
        np.linalg.norm(left, axis=0, keepdims=True), 1e-12
    )
    right_norm = right / np.maximum(
        np.linalg.norm(right, axis=0, keepdims=True), 1e-12
    )
    return np.einsum("ni,nj->ij", left_norm, right_norm, optimize=False)


def _validate_source_indices(source_indices: Iterable[int], width: int) -> list[int]:
    indices = sorted({int(value) for value in source_indices})
    if not indices or any(index < 0 or index >= width for index in indices):
        raise ValueError("source indices must be non-empty and fit the reference width")
    return indices


def _top_k(scores: np.ndarray, size: int) -> list[int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if size > len(values) or not np.all(np.isfinite(values)):
        raise ValueError("target scores must be finite and cover the requested subset")
    order = np.lexsort((np.arange(len(values)), -values))
    return sorted(int(value) for value in order[:size])


def _assignment_result(
    method: str,
    similarity: np.ndarray,
    source_indices: list[int],
    use_absolute: bool,
    metadata: dict | None = None,
) -> dict:
    objective = np.abs(similarity) if use_absolute else similarity
    source_order, target_order = _linear_sum_assignment(-objective)
    mapping = {
        int(source): int(target) for source, target in zip(source_order, target_order)
    }
    if any(index not in mapping for index in source_indices):
        raise ValueError("assignment does not cover every source index")
    target_indices = sorted(mapping[index] for index in source_indices)
    target_scores = np.zeros(similarity.shape[1], dtype=float)
    for source, target in mapping.items():
        if source in source_indices:
            target_scores[target] = float(objective[source, target])
    return {
        "method": method,
        "partition": "validation",
        "target_indices": target_indices,
        "target_scores": target_scores.tolist(),
        "metadata": {
            "fit_partition": "validation",
            "one_to_one": True,
            **(metadata or {}),
        },
    }


def _torch_matrix(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float64)


def _ridge_coefficients(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    left_tensor = _torch_matrix(left)
    right_tensor = _torch_matrix(right)
    gram = left_tensor.T @ left_tensor
    regularized = gram + float(alpha) * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    coefficients = torch.linalg.solve(regularized, left_tensor.T @ right_tensor)
    return coefficients.cpu().numpy()


def _ridge_map(
    left: np.ndarray,
    right: np.ndarray,
    alphas: tuple[float, ...],
    validation_fraction: float,
) -> tuple[np.ndarray, dict]:
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("blocked_validation_fraction must be in (0, 0.5)")
    split = int(np.floor(len(left) * (1.0 - float(validation_fraction))))
    split = min(max(split, 3), len(left) - 1)
    train_left, validation_left = left[:split], left[split:]
    train_right, validation_right = right[:split], right[split:]
    candidates = tuple(float(value) for value in alphas)
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("ridge_alphas must contain non-negative values")
    errors = []
    for alpha in candidates:
        coefficients = _ridge_coefficients(train_left, train_right, alpha)
        prediction = np.einsum(
            "ni,ij->nj", validation_left, coefficients, optimize=False
        )
        errors.append(float(np.mean((prediction - validation_right) ** 2)))
    best_index = int(np.argmin(errors))
    selected_alpha = candidates[best_index]
    return _ridge_coefficients(left, right, selected_alpha), {
        "selection_partition": "validation_tail",
        "fit_partition": "validation",
        "selected_alpha": selected_alpha,
        "validation_mse": errors[best_index],
    }


def _cca_map(
    left: np.ndarray,
    right: np.ndarray,
    max_components: int,
) -> tuple[np.ndarray, dict]:
    components = max(
        1,
        min(int(max_components), left.shape[1], right.shape[1], len(left) - 1),
    )
    left_tensor = _torch_matrix(left)
    right_tensor = _torch_matrix(right)
    denominator = float(max(len(left) - 1, 1))
    covariance_left = (left_tensor.T @ left_tensor) / denominator
    covariance_right = (right_tensor.T @ right_tensor) / denominator
    cross_covariance = (left_tensor.T @ right_tensor) / denominator
    ridge = 1e-6

    def inverse_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            matrix + ridge * torch.eye(matrix.shape[0], dtype=matrix.dtype)
        )
        inverse = torch.clamp(eigenvalues, min=ridge).rsqrt()
        return (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T

    left_inverse_sqrt = inverse_sqrt(covariance_left)
    right_inverse_sqrt = inverse_sqrt(covariance_right)
    whitened = left_inverse_sqrt @ cross_covariance @ right_inverse_sqrt
    source_vectors, canonical_values, _target_vectors = torch.linalg.svd(
        whitened, full_matrices=False
    )
    source_directions = left_inverse_sqrt @ source_vectors[:, :components]
    source_scores = left_tensor @ source_directions
    decoder = torch.linalg.lstsq(source_scores, right_tensor).solution
    coefficient_tensor = source_directions @ decoder
    reconstruction_tensor = left_tensor @ coefficient_tensor
    coefficients = coefficient_tensor.cpu().numpy()
    reconstruction = reconstruction_tensor.cpu().numpy()
    return coefficients, {
        "fit_partition": "validation",
        "components": components,
        "validation_reconstruction_mse": float(np.mean((reconstruction - right) ** 2)),
        "canonical_correlations": canonical_values[:components].cpu().tolist(),
    }


def _sinkhorn_plan(
    cost: np.ndarray,
    regularization: float,
    max_iterations: int = 1_000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int]:
    if regularization <= 0:
        raise ValueError("ot_regularization must be positive")
    rows, columns = cost.shape
    source_mass = np.full(rows, 1.0 / rows)
    target_mass = np.full(columns, 1.0 / columns)
    kernel = np.exp(-(cost - cost.min()) / float(regularization))
    kernel = np.maximum(kernel, 1e-300)
    left_scale = np.ones(rows)
    right_scale = np.ones(columns)
    for iteration in range(1, int(max_iterations) + 1):
        previous = left_scale.copy()
        left_product = np.einsum("ij,j->i", kernel, right_scale, optimize=False)
        left_scale = source_mass / np.maximum(left_product, 1e-300)
        right_product = np.einsum("ij,i->j", kernel, left_scale, optimize=False)
        right_scale = target_mass / np.maximum(right_product, 1e-300)
        if np.max(np.abs(left_scale - previous)) <= tolerance:
            break
    plan = left_scale[:, None] * kernel * right_scale[None, :]
    return plan, iteration


def _dense_result(
    method: str,
    coefficients: np.ndarray,
    source_indices: list[int],
    metadata: dict,
) -> dict:
    selected = coefficients[np.asarray(source_indices, dtype=int), :]
    target_scores = np.sqrt(np.sum(np.square(selected), axis=0))
    return {
        "method": method,
        "partition": "validation",
        "target_indices": _top_k(target_scores, len(source_indices)),
        "target_scores": target_scores.tolist(),
        "metadata": {"fit_partition": "validation", "one_to_one": False, **metadata},
    }


def fit_alignment_method(
    method: str,
    reference,
    target,
    source_indices: Iterable[int],
    *,
    random_state: int = 1,
    reference_weight_signatures=None,
    target_weight_signatures=None,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0, 100.0),
    blocked_validation_fraction: float = 0.25,
    cca_components: int = 16,
    ot_regularization: float = 0.05,
) -> dict:
    del random_state  # All current fits are deterministic; retained in the public API.
    method = str(method)
    if method not in ALIGNMENT_METHODS:
        raise ValueError(f"unknown alignment method: {method}")
    left, right = _paired_matrices(reference, target)
    indices = _validate_source_indices(source_indices, left.shape[1])
    if method == "identity":
        if right.shape[1] <= max(indices):
            raise ValueError("source indices do not fit the target width")
        scores = np.zeros(right.shape[1], dtype=float)
        scores[indices] = 1.0
        return {
            "method": method,
            "partition": "validation",
            "target_indices": indices,
            "target_scores": scores.tolist(),
            "metadata": {
                "fit_partition": "validation",
                "one_to_one": True,
                "identity_control": True,
            },
        }
    if method in {"hungarian_abs", "hungarian_signed"}:
        correlation = _correlation_matrix(left, right)
        return _assignment_result(
            method,
            correlation,
            indices,
            use_absolute=method == "hungarian_abs",
            metadata={"similarity": "activation_correlation"},
        )
    if method == "procrustes":
        if left.shape[1] != right.shape[1]:
            raise ValueError("procrustes requires equal representation widths")
        left_tensor = _torch_matrix(left)
        right_tensor = _torch_matrix(right)
        source_vectors, singular_values, target_vectors = torch.linalg.svd(
            left_tensor.T @ right_tensor,
            full_matrices=False,
        )
        rotation = (source_vectors @ target_vectors).cpu().numpy()
        return _dense_result(
            method,
            rotation,
            indices,
            {"orthogonal_scale": float(singular_values.sum().cpu())},
        )
    if method == "cca":
        coefficients, metadata = _cca_map(left, right, cca_components)
        return _dense_result(method, coefficients, indices, metadata)
    if method == "optimal_transport":
        correlation = _correlation_matrix(left, right)
        plan, iterations = _sinkhorn_plan(
            1.0 - np.abs(correlation), regularization=ot_regularization
        )
        target_scores = plan[np.asarray(indices, dtype=int), :].sum(axis=0)
        return {
            "method": method,
            "partition": "validation",
            "target_indices": _top_k(target_scores, len(indices)),
            "target_scores": target_scores.tolist(),
            "metadata": {
                "fit_partition": "validation",
                "one_to_one": False,
                "regularization": float(ot_regularization),
                "iterations": int(iterations),
            },
        }
    if method == "weight_matching":
        if reference_weight_signatures is None or target_weight_signatures is None:
            raise ValueError("weight_matching requires frozen adjacent weight signatures")
        reference_signatures = _standardize(reference_weight_signatures)
        target_signatures = _standardize(target_weight_signatures)
        if reference_signatures.shape[0] != left.shape[1] or target_signatures.shape[0] != right.shape[1]:
            raise ValueError("weight signatures must provide one row per hidden unit")
        similarity = _correlation_matrix(
            reference_signatures.T,
            target_signatures.T,
        )
        return _assignment_result(
            method,
            similarity,
            indices,
            use_absolute=False,
            metadata={"uses_frozen_weights": True, "similarity": "adjacent_weight_cosine"},
        )
    coefficients, metadata = _ridge_map(
        left,
        right,
        alphas=ridge_alphas,
        validation_fraction=blocked_validation_fraction,
    )
    return _dense_result(method, coefficients, indices, metadata)


def circular_shift_alignment_controls(
    reference,
    target,
    source_indices: Iterable[int],
    repeats: int,
    random_state: int = 1,
) -> list[dict]:
    """Refit Hungarian alignment after nonzero circular shifts of validation rows."""
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    left, right = _paired_matrices(reference, target)
    if len(left) < 2:
        raise ValueError("circular-shift alignment controls require at least two rows")
    rng = np.random.default_rng(random_state)
    shifts = rng.integers(1, len(left), size=int(repeats))
    controls = []
    for shift in shifts:
        fitted = fit_alignment_method(
            "hungarian_abs",
            left,
            np.roll(right, int(shift), axis=0),
            source_indices,
        )
        controls.append(
            {
                "shift": int(shift),
                "target_indices": [int(value) for value in fitted["target_indices"]],
                "metadata": {
                    **fitted["metadata"],
                    "fit_partition": "validation_circular_shift_null",
                },
            }
        )
    return controls


def blocked_validation_alignment_stability(
    reference,
    target,
    source_indices: Iterable[int],
    blocks: int = 4,
) -> dict:
    """Audit Hungarian mappings under contiguous leave-one-block-out refits."""
    left, right = _paired_matrices(reference, target)
    if int(blocks) < 2:
        raise ValueError("blocks must be at least two")
    block_count = min(int(blocks), len(left))
    if len(left) - int(np.ceil(len(left) / block_count)) < 2:
        raise ValueError("blocked stability requires at least two retained rows")
    full = fit_alignment_method(
        "hungarian_abs",
        left,
        right,
        source_indices,
    )
    full_indices = [int(value) for value in full["target_indices"]]
    full_set = set(full_indices)
    perturbations = []
    for block_index, excluded in enumerate(
        np.array_split(np.arange(len(left), dtype=int), block_count)
    ):
        keep = np.ones(len(left), dtype=bool)
        keep[excluded] = False
        fitted = fit_alignment_method(
            "hungarian_abs",
            left[keep],
            right[keep],
            source_indices,
        )
        indices = [int(value) for value in fitted["target_indices"]]
        selected = set(indices)
        union = full_set | selected
        perturbations.append(
            {
                "block_index": int(block_index),
                "excluded_start": int(excluded[0]),
                "excluded_end": int(excluded[-1]),
                "target_indices": indices,
                "exact_match": float(selected == full_set),
                "jaccard": float(len(selected & full_set) / len(union)),
            }
        )
    return {
        "method": "hungarian_abs",
        "perturbation": "contiguous_leave_one_validation_block_out",
        "blocks": int(block_count),
        "full_target_indices": full_indices,
        "exact_match_rate": float(
            np.mean([row["exact_match"] for row in perturbations])
        ),
        "mean_jaccard": float(np.mean([row["jaccard"] for row in perturbations])),
        "perturbations": perturbations,
    }
