from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
from torch import nn

from TradingCircuitDiscovery.evaluation.causality_test import (
    compare_action_score_effect,
)
from TradingCircuitDiscovery.evaluation.representation_stability import (
    hungarian_neuron_alignment,
)
from TradingCircuitDiscovery.intervention.activation_patch import resolve_module


def _linear_module(network: nn.Module, name: str) -> nn.Linear:
    module = resolve_module(network, name)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{name!r} must resolve to torch.nn.Linear")
    return module


def _permutation_tensor(indices: Sequence[int], width: int, device) -> torch.Tensor:
    values = [int(index) for index in indices]
    if sorted(values) != list(range(width)):
        raise ValueError(f"indices must be a complete permutation of 0..{width - 1}")
    return torch.tensor(values, dtype=torch.long, device=device)


def permute_hidden_units(
    network: nn.Module,
    *,
    linear_layers: Sequence[str],
    target_to_source: Sequence[Sequence[int]],
) -> nn.Module:
    """Return a function-preserving copy with permuted ReLU hidden units.

    ``linear_layers`` lists every linear layer from input to output.
    ``target_to_source[k][j]`` is the source-unit index represented by target
    unit ``j`` at hidden layer ``k``.
    """

    names = tuple(str(name) for name in linear_layers)
    permutations = tuple(
        tuple(int(index) for index in values) for values in target_to_source
    )
    if len(names) < 2 or len(permutations) != len(names) - 1:
        raise ValueError(
            "one target-to-source permutation is required per hidden layer"
        )

    source_layers = tuple(_linear_module(network, name) for name in names)
    target = deepcopy(network)
    target_layers = tuple(_linear_module(target, name) for name in names)

    with torch.no_grad():
        previous: torch.Tensor | None = None
        for source, destination, values in zip(
            source_layers[:-1], target_layers[:-1], permutations
        ):
            if source.out_features != destination.out_features:
                raise ValueError("source and copied hidden widths differ")
            permutation = _permutation_tensor(
                values,
                source.out_features,
                source.weight.device,
            )
            weight = source.weight.index_select(0, permutation)
            if previous is not None:
                weight = weight.index_select(1, previous)
            destination.weight.copy_(weight)
            if source.bias is not None:
                if destination.bias is None:
                    raise ValueError("source and copied bias settings differ")
                destination.bias.copy_(source.bias.index_select(0, permutation))
            previous = permutation

        source_output = source_layers[-1]
        target_output = target_layers[-1]
        if previous is None:
            raise ValueError("at least one hidden layer is required")
        target_output.weight.copy_(source_output.weight.index_select(1, previous))
        if source_output.bias is not None:
            if target_output.bias is None:
                raise ValueError("source and copied output-bias settings differ")
            target_output.bias.copy_(source_output.bias)

    target.eval()
    target._known_hidden_permutations = {
        name: {
            target_index: source_index
            for target_index, source_index in enumerate(values)
        }
        for name, values in zip(names[:-1], permutations)
    }
    return target


def capture_pre_activations(
    network: nn.Module,
    states: torch.Tensor,
    hidden_layers: Sequence[str],
) -> dict[str, torch.Tensor]:
    storage: dict[str, torch.Tensor] = {}
    handles = []

    for name in hidden_layers:
        module = _linear_module(network, str(name))

        def make_hook(layer_name: str):
            def hook(_module, _inputs, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("linear-layer output must be a tensor")
                storage[layer_name] = output.detach().cpu()

            return hook

        handles.append(module.register_forward_hook(make_hook(str(name))))

    parameter = next(network.parameters())
    state_tensor = torch.as_tensor(
        states,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    was_training = network.training
    try:
        network.eval()
        with torch.no_grad():
            network(state_tensor)
    finally:
        network.train(was_training)
        for handle in handles:
            handle.remove()
    return storage


def _network_outputs(network: nn.Module, states: torch.Tensor) -> torch.Tensor:
    parameter = next(network.parameters())
    state_tensor = torch.as_tensor(
        states,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    was_training = network.training
    try:
        network.eval()
        with torch.no_grad():
            return network(state_tensor).detach().cpu()
    finally:
        network.train(was_training)


def recover_hidden_mappings(
    reference: nn.Module,
    target: nn.Module,
    states: torch.Tensor,
    *,
    hidden_layers: Sequence[str],
) -> dict[str, dict[int, int]]:
    reference_values = capture_pre_activations(reference, states, hidden_layers)
    target_values = capture_pre_activations(target, states, hidden_layers)
    return {
        name: hungarian_neuron_alignment(
            reference_values[name],
            target_values[name],
        )["target_to_reference"]
        for name in hidden_layers
    }


def _intervened_outputs(
    network: nn.Module,
    states: torch.Tensor,
    *,
    hidden_layer: str,
    indices: Sequence[int],
    values: torch.Tensor,
) -> torch.Tensor:
    module = _linear_module(network, hidden_layer)
    selected = [int(index) for index in indices]
    parameter = next(network.parameters())
    replacement = torch.as_tensor(
        values,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    state_tensor = torch.as_tensor(
        states,
        dtype=parameter.dtype,
        device=parameter.device,
    )

    def hook(_module, _inputs, output):
        edited = output.clone()
        edited[..., selected] = replacement
        return edited

    handle = module.register_forward_hook(hook)
    was_training = network.training
    try:
        network.eval()
        with torch.no_grad():
            return network(state_tensor).detach().cpu()
    finally:
        network.train(was_training)
        handle.remove()


def _target_indices(
    source_indices: Sequence[int],
    target_to_source: dict[int, int],
) -> list[int]:
    selected = {int(index) for index in source_indices}
    mapped = sorted(
        target for target, source in target_to_source.items() if source in selected
    )
    if len(mapped) != len(selected):
        raise ValueError("mapping does not cover every selected source unit")
    return mapped


def _random_control_mapping(
    expected: dict[int, int],
    source_indices: Sequence[int],
    rng: np.random.Generator,
) -> dict[int, int]:
    """Sample a permutation whose intervention subset differs from the truth."""

    width = len(expected)
    selected_sources = {int(index) for index in source_indices}
    if not selected_sources or len(selected_sources) >= width:
        raise ValueError(
            "random control requires a non-empty strict hidden-unit subset"
        )

    expected_values = [int(expected[index]) for index in range(width)]
    mapped_targets = frozenset(_target_indices(selected_sources, expected))
    for _ in range(1024):
        random_values = rng.permutation(width).tolist()
        candidate = {
            target_index: int(source_index)
            for target_index, source_index in enumerate(random_values)
        }
        if frozenset(_target_indices(selected_sources, candidate)) != mapped_targets:
            return candidate

    # Deterministic fallback: swapping one selected and one unselected target
    # keeps a valid permutation while forcing a different intervention subset.
    selected_target = int(rng.choice(sorted(mapped_targets)))
    unselected_target = int(
        rng.choice([index for index in range(width) if index not in mapped_targets])
    )
    expected_values[selected_target], expected_values[unselected_target] = (
        expected_values[unselected_target],
        expected_values[selected_target],
    )
    return {
        target_index: source_index
        for target_index, source_index in enumerate(expected_values)
    }


def _effect(
    original: torch.Tensor,
    intervened: torch.Tensor,
    *,
    probability_scores: bool,
) -> float:
    result = compare_action_score_effect(
        original.detach().cpu().numpy(),
        intervened.detach().cpu().numpy(),
        probability_scores=probability_scores,
    )
    return float(result["mean_absolute_margin_change"])


def _prepare_known_permutation(
    reference: nn.Module,
    target: nn.Module,
    *,
    discovery_states: torch.Tensor,
    validation_states: torch.Tensor,
    test_states: torch.Tensor,
    hidden_layer: str,
    source_indices: Sequence[int],
    probability_scores: bool,
) -> dict:
    known_by_layer = getattr(target, "_known_hidden_permutations", None)
    if not isinstance(known_by_layer, dict) or hidden_layer not in known_by_layer:
        raise ValueError("target does not carry known hidden-permutation metadata")
    expected = known_by_layer[hidden_layer]

    recovered = recover_hidden_mappings(
        reference,
        target,
        validation_states,
        hidden_layers=(hidden_layer,),
    )[hidden_layer]
    width = len(expected)
    mapping_accuracy = float(
        np.mean([recovered.get(index) == expected[index] for index in range(width)])
    )

    reference_discovery = capture_pre_activations(
        reference,
        discovery_states,
        (hidden_layer,),
    )[hidden_layer]
    target_discovery = capture_pre_activations(
        target,
        discovery_states,
        (hidden_layer,),
    )[hidden_layer]
    source_selected = [int(index) for index in source_indices]
    if len(set(source_selected)) != len(source_selected):
        raise ValueError("source_indices must not contain duplicates")
    mapped_selected = _target_indices(source_selected, recovered)

    reference_original = _network_outputs(reference, test_states)
    target_original = _network_outputs(target, test_states)
    reference_intervened = _intervened_outputs(
        reference,
        test_states,
        hidden_layer=hidden_layer,
        indices=source_selected,
        values=reference_discovery[:, source_selected].mean(dim=0),
    )
    target_intervened = _intervened_outputs(
        target,
        test_states,
        hidden_layer=hidden_layer,
        indices=mapped_selected,
        values=target_discovery[:, mapped_selected].mean(dim=0),
    )
    source_effect = _effect(
        reference_original,
        reference_intervened,
        probability_scores=probability_scores,
    )
    mapped_effect = _effect(
        target_original,
        target_intervened,
        probability_scores=probability_scores,
    )
    output_scale = float(
        max(
            torch.max(torch.abs(reference_original)).item(),
            torch.max(torch.abs(target_original)).item(),
        )
    )
    mapped_intervened_output_scale = float(
        max(
            torch.max(torch.abs(reference_intervened)).item(),
            torch.max(torch.abs(target_intervened)).item(),
        )
    )

    return {
        "expected": expected,
        "recovered": recovered,
        "width": width,
        "source_selected": source_selected,
        "target_discovery": target_discovery,
        "reference_intervened": reference_intervened,
        "target_original": target_original,
        "fixed": {
            "mapping_accuracy": mapping_accuracy,
            "numeric_dtype": str(reference_original.dtype),
            "output_scale": output_scale,
            "mapped_intervened_output_scale": mapped_intervened_output_scale,
            "max_output_error": float(
                torch.max(torch.abs(reference_original - target_original)).item()
            ),
            "mapped_intervened_output_error": float(
                torch.max(torch.abs(reference_intervened - target_intervened)).item()
            ),
            "source_effect": source_effect,
            "mapped_effect": mapped_effect,
            "mapped_effect_error": abs(mapped_effect - source_effect),
        },
    }


def _evaluate_random_control(
    target: nn.Module,
    prepared: dict,
    *,
    test_states: torch.Tensor,
    hidden_layer: str,
    probability_scores: bool,
    random_seed: int,
) -> dict[str, float]:
    expected = prepared["expected"]
    recovered = prepared["recovered"]
    width = int(prepared["width"])
    source_selected = prepared["source_selected"]
    rng = np.random.default_rng(int(random_seed))
    random_mapping = _random_control_mapping(recovered, source_selected, rng)
    random_selected = _target_indices(source_selected, random_mapping)
    target_discovery = prepared["target_discovery"]
    random_intervened = _intervened_outputs(
        target,
        test_states,
        hidden_layer=hidden_layer,
        indices=random_selected,
        values=target_discovery[:, random_selected].mean(dim=0),
    )
    target_original = prepared["target_original"]
    random_effect = _effect(
        target_original,
        random_intervened,
        probability_scores=probability_scores,
    )
    source_effect = float(prepared["fixed"]["source_effect"])

    return {
        **prepared["fixed"],
        "random_mapping_accuracy": float(
            np.mean(
                [random_mapping[index] == expected[index] for index in range(width)]
            )
        ),
        "random_intervened_output_error": float(
            torch.max(
                torch.abs(prepared["reference_intervened"] - random_intervened)
            ).item()
        ),
        "random_effect": random_effect,
        "random_effect_error": abs(random_effect - source_effect),
    }


def evaluate_known_permutation_controls(
    reference: nn.Module,
    target: nn.Module,
    *,
    discovery_states: torch.Tensor,
    validation_states: torch.Tensor,
    test_states: torch.Tensor,
    hidden_layer: str,
    source_indices: Sequence[int],
    probability_scores: bool,
    random_seeds: Sequence[int],
) -> list[dict[str, float]]:
    controls = tuple(int(seed) for seed in random_seeds)
    if not controls:
        raise ValueError("at least one random control seed is required")
    prepared = _prepare_known_permutation(
        reference,
        target,
        discovery_states=discovery_states,
        validation_states=validation_states,
        test_states=test_states,
        hidden_layer=hidden_layer,
        source_indices=source_indices,
        probability_scores=probability_scores,
    )
    return [
        _evaluate_random_control(
            target,
            prepared,
            test_states=test_states,
            hidden_layer=hidden_layer,
            probability_scores=probability_scores,
            random_seed=seed,
        )
        for seed in controls
    ]


def evaluate_known_permutation(
    reference: nn.Module,
    target: nn.Module,
    *,
    discovery_states: torch.Tensor,
    validation_states: torch.Tensor,
    test_states: torch.Tensor,
    hidden_layer: str,
    source_indices: Sequence[int],
    probability_scores: bool,
    random_seed: int,
) -> dict[str, float]:
    return evaluate_known_permutation_controls(
        reference,
        target,
        discovery_states=discovery_states,
        validation_states=validation_states,
        test_states=test_states,
        hidden_layer=hidden_layer,
        source_indices=source_indices,
        probability_scores=probability_scores,
        random_seeds=(int(random_seed),),
    )[0]
