from __future__ import annotations

import numpy as np
import torch


def _activation_vectors_by_key(bank, layer: str, partition: str | None) -> dict[str, np.ndarray]:
    values = {}
    for index, record in enumerate(bank.records):
        if partition is not None and record.get("partition") != partition:
            continue
        if layer not in record.get("activations", {}):
            continue
        activation = record["activations"][layer]
        if hasattr(activation, "detach"):
            activation = activation.detach().cpu().numpy()
        key = str(record.get("date", index))
        values[key] = np.asarray(activation, dtype=np.float64).reshape(-1)
    return values


def paired_activation_matrices(
    reference_bank,
    target_bank,
    layer: str,
    partition: str | None = "validation",
) -> tuple[np.ndarray, np.ndarray]:
    reference = _activation_vectors_by_key(reference_bank, layer, partition)
    target = _activation_vectors_by_key(target_bank, layer, partition)
    common = sorted(set(reference) & set(target))
    if len(common) < 2:
        raise ValueError("alignment requires at least two common dated observations")
    left = np.vstack([reference[key] for key in common])
    right = np.vstack([target[key] for key in common])
    if left.shape[1] != right.shape[1]:
        raise ValueError("alignment requires equal representation widths")
    return left, right


def _center(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 2:
        raise ValueError("representation matrices must be two-dimensional with at least two rows")
    return array - array.mean(axis=0, keepdims=True)


def _linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost matrix must be two-dimensional")
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    rows, columns = cost.shape
    row_potential = np.zeros(rows + 1)
    column_potential = np.zeros(columns + 1)
    matching = np.zeros(columns + 1, dtype=int)
    predecessor = np.zeros(columns + 1, dtype=int)
    for row in range(1, rows + 1):
        matching[0] = row
        column = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column] = True
            matched_row = matching[column]
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, columns + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    cost[matched_row - 1, candidate_column - 1]
                    - row_potential[matched_row]
                    - column_potential[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(columns + 1):
                if used[candidate_column]:
                    row_potential[matching[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matching[column] = matching[previous_column]
            column = previous_column
            if column == 0:
                break
    assigned_rows = []
    assigned_columns = []
    for column in range(1, columns + 1):
        if matching[column] != 0:
            assigned_rows.append(matching[column] - 1)
            assigned_columns.append(column - 1)
    assigned_rows = np.asarray(assigned_rows, dtype=int)
    assigned_columns = np.asarray(assigned_columns, dtype=int)
    order = np.argsort(assigned_rows)
    assigned_rows = assigned_rows[order]
    assigned_columns = assigned_columns[order]
    if transposed:
        return assigned_columns, assigned_rows
    return assigned_rows, assigned_columns


def hungarian_neuron_alignment(reference, target) -> dict:
    left = _center(reference)
    right = _center(target)
    if len(left) != len(right):
        raise ValueError("representations must share aligned observations")
    left_tensor = torch.from_numpy(left)
    right_tensor = torch.from_numpy(right)
    left_tensor = left_tensor / torch.clamp(torch.linalg.vector_norm(left_tensor, dim=0, keepdim=True), min=1e-12)
    right_tensor = right_tensor / torch.clamp(torch.linalg.vector_norm(right_tensor, dim=0, keepdim=True), min=1e-12)
    correlations = (left_tensor.T @ right_tensor).cpu().numpy()
    reference_indices, target_indices = _linear_sum_assignment(-np.abs(correlations))
    mapping = {int(target_index): int(reference_index) for reference_index, target_index in zip(reference_indices, target_indices)}
    matched = [abs(correlations[reference_index, target_index]) for reference_index, target_index in zip(reference_indices, target_indices)]
    return {
        "target_to_reference": mapping,
        "mean_absolute_correlation": float(np.mean(matched)) if matched else float("nan"),
        "min_absolute_correlation": float(np.min(matched)) if matched else float("nan"),
    }


def aligned_jaccard(reference_indices, target_indices, target_to_reference: dict[int, int]) -> float:
    left = {int(value) for value in reference_indices}
    right = {target_to_reference[int(value)] for value in target_indices if int(value) in target_to_reference}
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def linear_cka(reference, target) -> float:
    left = torch.from_numpy(_center(reference))
    right = torch.from_numpy(_center(target))
    numerator = torch.linalg.matrix_norm(left.T @ right, ord="fro") ** 2
    denominator = torch.linalg.matrix_norm(left.T @ left, ord="fro") * torch.linalg.matrix_norm(right.T @ right, ord="fro")
    return float((numerator / torch.clamp(denominator, min=1e-12)).cpu())


def orthogonal_procrustes_residual(reference, target) -> float:
    left = torch.from_numpy(_center(reference))
    right = torch.from_numpy(_center(target))
    left = left / torch.clamp(torch.linalg.matrix_norm(left, ord="fro"), min=1e-12)
    right = right / torch.clamp(torch.linalg.matrix_norm(right, ord="fro"), min=1e-12)
    left_vectors, _singular_values, right_vectors = torch.linalg.svd(right.T @ left, full_matrices=False)
    rotation = left_vectors @ right_vectors
    return float(torch.linalg.matrix_norm(left - right @ rotation, ord="fro").cpu())
