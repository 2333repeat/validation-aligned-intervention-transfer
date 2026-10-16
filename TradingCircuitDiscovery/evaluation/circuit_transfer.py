from __future__ import annotations


def map_reference_circuit_to_target(
    reference_indices,
    target_to_reference: dict[int, int],
) -> list[int]:
    requested = {int(index) for index in reference_indices}
    mapped = sorted(
        int(target_index)
        for target_index, reference_index in target_to_reference.items()
        if int(reference_index) in requested
    )
    covered = {
        int(target_to_reference[target_index])
        for target_index in mapped
    }
    if covered != requested:
        missing = sorted(requested - covered)
        raise ValueError(
            "alignment does not cover every source neuron; "
            f"missing reference indices: {missing}"
        )
    return mapped
