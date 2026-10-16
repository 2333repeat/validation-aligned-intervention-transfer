from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from TradingCircuitDiscovery.collect.dataset import ActivationBank
from TradingCircuitDiscovery.evaluation.causality_test import (
    action_score_observation_effects,
    compare_action_score_effect,
    compare_transfer_profiles,
    matched_random_circuits,
)
from TradingCircuitDiscovery.evaluation.alignment_methods import (
    ALIGNMENT_METHODS,
    blocked_validation_alignment_stability,
    checkpoint_weight_signatures,
    circular_shift_alignment_controls,
    fit_alignment_method,
)
from TradingCircuitDiscovery.evaluation.representation_stability import (
    hungarian_neuron_alignment,
    paired_activation_matrices,
)
from TradingCircuitDiscovery.experiments.manifest import ExperimentRun
from TradingCircuitDiscovery.experiments.run_checkpoint import (
    _default_layers,
    _distributions,
    _write_json,
    run_directory,
)
from TradingCircuitDiscovery.integrations.policy_value import build_policy_value_adapter


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _algorithm_config(algorithm: str) -> str:
    return f"configs/{algorithm}.yaml"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _supplement_path(root: str | Path, run: ExperimentRun) -> Path:
    return run_directory(root, run) / "supplement.json"


def _activation_path(root: str | Path, run: ExperimentRun) -> Path:
    return run_directory(root, run) / "activation" / "oos.pt"


def _layer_matrix(bank: ActivationBank, layer: str) -> np.ndarray:
    rows = []
    for record in bank.records:
        activation = record.get("activations", {}).get(layer)
        if activation is None:
            raise KeyError(f"activation bank is missing layer {layer!r}")
        if hasattr(activation, "detach"):
            activation = activation.detach().cpu().numpy()
        rows.append(np.asarray(activation, dtype=np.float32).reshape(-1))
    return np.vstack(rows)


def _effect(original, intervened) -> dict[str, float]:
    return compare_action_score_effect(
        original,
        intervened,
        probability_scores=True,
    )


def _numeric_profile(profile: dict) -> dict[str, float | None]:
    return {
        key: value
        for key, value in profile.items()
        if not key.endswith("transition_counts")
    }


def _mean_nested_metrics(rows: list[dict], key: str) -> dict[str, float | None]:
    metric_names = sorted({name for row in rows for name in row.get(key, {})})
    output = {}
    for metric in metric_names:
        values = [
            float(row[key][metric])
            for row in rows
            if row.get(key, {}).get(metric) is not None
            and np.isfinite(float(row[key][metric]))
        ]
        output[metric] = float(np.mean(values)) if values else None
    return output


def _transfer_random_state(
    source_run: ExperimentRun,
    target_run: ExperimentRun,
    concept: str,
    namespace: str,
) -> int:
    identity = "|".join(
        (
            str(source_run.run_id),
            str(target_run.run_id),
            str(concept),
            str(namespace),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def run_transfer_checkpoint(
    target_run: ExperimentRun,
    source_runs: list[ExperimentRun],
    project_root: str | Path,
    base_output_root: str | Path,
    supplement_output_root: str | Path,
    output_root: str | Path,
    random_controls: int = 50,
    alignment_null_controls: int | None = None,
    alignment_methods: tuple[str, ...] = ("hungarian_abs",),
    adapter=None,
) -> dict:
    if int(random_controls) <= 0:
        raise ValueError("random_controls must be positive")
    null_repeats = (
        int(random_controls)
        if alignment_null_controls is None
        else int(alignment_null_controls)
    )
    if null_repeats <= 0:
        raise ValueError("alignment_null_controls must be positive")
    methods = tuple(dict.fromkeys(str(value) for value in alignment_methods))
    unknown_methods = set(methods) - set(ALIGNMENT_METHODS)
    if not methods or unknown_methods:
        raise ValueError(f"unknown alignment methods: {sorted(unknown_methods)}")
    output_dir = run_directory(output_root, target_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    started_at = _utc_now()
    _write_json(
        status_path,
        {"run_id": target_run.run_id, "status": "running", "started_at": started_at},
    )
    try:
        target_supplement = _read_json(
            _supplement_path(supplement_output_root, target_run)
        )
        target_bank = ActivationBank.load(_activation_path(base_output_root, target_run))
        if adapter is None:
            adapter = build_policy_value_adapter(
                algorithm=target_run.algorithm,
                agent_config_path=_algorithm_config(target_run.algorithm),
                checkpoint_path=target_run.checkpoint_path,
                layers=_default_layers(target_run.algorithm),
                project_root=project_root,
            )
        partitions = np.asarray(
            [record.get("partition", "excluded") for record in target_bank.records]
        )
        test_mask = partitions == "test"
        discovery_mask = partitions == "discovery"
        if int(test_mask.sum()) == 0 or int(discovery_mask.sum()) == 0:
            raise ValueError("target activation bank requires discovery and test rows")
        states = np.vstack(
            [np.asarray(record["state"], dtype=np.float32) for record in target_bank.records]
        )
        test_states = states[test_mask]
        test_dates = [
            str(record.get("date"))
            for record, selected in zip(target_bank.records, test_mask)
            if selected
        ]
        if any(value in {"", "None"} for value in test_dates):
            raise ValueError("target activation bank requires dated test rows")
        baseline_test = _distributions(adapter, test_states)
        layer_cache = {}
        target_weight_signature_cache = {}
        source_payload_cache = {}
        source_bank_cache = {}
        source_weight_signature_cache = {}
        rows = []
        for concept, target_concept in target_supplement.get("concepts", {}).items():
            if target_concept.get("status") != "completed":
                continue
            target_layer = target_concept["selected_layer"]
            if target_layer not in layer_cache:
                layer_cache[target_layer] = _layer_matrix(target_bank, target_layer)
            target_values = layer_cache[target_layer]
            target_mean = target_values[discovery_mask].mean(axis=0)
            native_indices = [
                int(value)
                for value in target_concept["final_circuit"]["indices"]
            ]
            native_intervened = _distributions(
                adapter,
                test_states,
                layer=target_layer,
                indices=native_indices,
                values=target_mean[native_indices],
            )
            native_effect = _effect(baseline_test, native_intervened)
            for source_run in source_runs:
                base = {
                    "algorithm": target_run.algorithm,
                    "market": target_run.market,
                    "concept": concept,
                    "source_run_id": source_run.run_id,
                    "source_seed": int(source_run.seed),
                    "target_run_id": target_run.run_id,
                    "target_seed": int(target_run.seed),
                    "alignment_partition": "validation",
                    "evaluation_partition": "test",
                    "target_layer": target_layer,
                }
                if (
                    source_run.algorithm != target_run.algorithm
                    or source_run.market != target_run.market
                    or source_run.seed == target_run.seed
                ):
                    rows.append(
                        {
                            **base,
                            "status": "skipped_invalid_source",
                            "error": "source must share algorithm/market and use a different seed",
                        }
                    )
                    continue
                try:
                    if source_run.run_id not in source_payload_cache:
                        source_payload_cache[source_run.run_id] = _read_json(
                            _supplement_path(supplement_output_root, source_run)
                        )
                    source_payload = source_payload_cache[source_run.run_id]
                    source_concept = source_payload.get("concepts", {}).get(concept)
                    if not source_concept or source_concept.get("status") != "completed":
                        rows.append({**base, "status": "skipped_source_concept"})
                        continue
                    source_layer = source_concept["selected_layer"]
                    base["source_layer"] = source_layer
                    if source_layer != target_layer:
                        rows.append({**base, "status": "skipped_layer_mismatch"})
                        continue
                    if source_run.run_id not in source_bank_cache:
                        source_bank_cache[source_run.run_id] = ActivationBank.load(
                            _activation_path(base_output_root, source_run)
                        )
                    reference, target = paired_activation_matrices(
                        source_bank_cache[source_run.run_id],
                        target_bank,
                        layer=target_layer,
                        partition="validation",
                    )
                    alignment = hungarian_neuron_alignment(reference, target)
                    source_indices = [
                        int(value)
                        for value in source_concept["final_circuit"]["indices"]
                    ]
                    width = int(target_values.shape[1])
                    if any(index < 0 or index >= width for index in source_indices):
                        raise ValueError("raw source indices do not fit the target representation")
                    raw_distribution = _distributions(
                        adapter,
                        test_states,
                        layer=target_layer,
                        indices=source_indices,
                        values=target_mean[source_indices],
                    )
                    raw_effect = _effect(baseline_test, raw_distribution)
                    raw_profile = compare_transfer_profiles(
                        baseline_test,
                        raw_distribution,
                        native_intervened,
                        probability_scores=True,
                    )
                    control_indices = matched_random_circuits(
                        width=width,
                        size=len(source_indices),
                        repeats=int(random_controls),
                        excluded=[],
                        random_state=_transfer_random_state(
                            source_run,
                            target_run,
                            concept,
                            "membership_control",
                        ),
                    )
                    controls = []
                    for indices in control_indices:
                        distribution = _distributions(
                            adapter,
                            test_states,
                            layer=target_layer,
                            indices=indices,
                            values=target_mean[indices],
                        )
                        controls.append(
                            {
                                "indices": [int(value) for value in indices],
                                "effect": _effect(baseline_test, distribution),
                                "profile_to_native": _numeric_profile(
                                    compare_transfer_profiles(
                                        baseline_test,
                                        distribution,
                                        native_intervened,
                                        probability_scores=True,
                                    )
                                ),
                            }
                        )
                    mean_control_effect = _mean_nested_metrics(controls, "effect")
                    mean_control_profile = _mean_nested_metrics(
                        controls, "profile_to_native"
                    )
                    absolute_control_effects = [
                        float(control["effect"]["mean_absolute_margin_change"])
                        for control in controls
                    ]
                    alignment_null_fits = circular_shift_alignment_controls(
                        reference,
                        target,
                        source_indices=source_indices,
                        repeats=null_repeats,
                        random_state=_transfer_random_state(
                            source_run,
                            target_run,
                            concept,
                            "alignment_null",
                        ),
                    )
                    alignment_null_rows = []
                    for null_fit in alignment_null_fits:
                        indices = [
                            int(value) for value in null_fit["target_indices"]
                        ]
                        distribution = _distributions(
                            adapter,
                            test_states,
                            layer=target_layer,
                            indices=indices,
                            values=target_mean[indices],
                        )
                        alignment_null_rows.append(
                            {
                                "shift": int(null_fit["shift"]),
                                "indices": indices,
                                "effect": _effect(baseline_test, distribution),
                                "profile_to_native": _numeric_profile(
                                    compare_transfer_profiles(
                                        baseline_test,
                                        distribution,
                                        native_intervened,
                                        probability_scores=True,
                                    )
                                ),
                            }
                        )
                    mean_alignment_null_effect = _mean_nested_metrics(
                        alignment_null_rows,
                        "effect",
                    )
                    mean_alignment_null_profile = _mean_nested_metrics(
                        alignment_null_rows,
                        "profile_to_native",
                    )
                    absolute_alignment_null_effects = [
                        float(row["effect"]["mean_absolute_margin_change"])
                        for row in alignment_null_rows
                    ]
                    mapping_stability = blocked_validation_alignment_stability(
                        reference,
                        target,
                        source_indices=source_indices,
                        blocks=4,
                    )
                    for method in methods:
                        reference_signatures = None
                        target_signatures = None
                        if method == "weight_matching":
                            source_signature_key = (source_run.run_id, source_layer)
                            if source_signature_key not in source_weight_signature_cache:
                                source_weight_signature_cache[source_signature_key] = (
                                    checkpoint_weight_signatures(
                                        source_run.checkpoint_path, source_layer
                                    )
                                )
                            if target_layer not in target_weight_signature_cache:
                                target_weight_signature_cache[target_layer] = (
                                    checkpoint_weight_signatures(
                                        target_run.checkpoint_path, target_layer
                                    )
                                )
                            reference_signatures = source_weight_signature_cache[
                                source_signature_key
                            ]
                            target_signatures = target_weight_signature_cache[target_layer]
                        fitted = fit_alignment_method(
                            method,
                            reference,
                            target,
                            source_indices,
                            random_state=_transfer_random_state(
                                source_run,
                                target_run,
                                concept,
                                f"alignment:{method}",
                            ),
                            reference_weight_signatures=reference_signatures,
                            target_weight_signatures=target_signatures,
                        )
                        aligned_indices = [
                            int(value) for value in fitted["target_indices"]
                        ]
                        aligned_distribution = _distributions(
                            adapter,
                            test_states,
                            layer=target_layer,
                            indices=aligned_indices,
                            values=target_mean[aligned_indices],
                        )
                        aligned_effect = _effect(
                            baseline_test, aligned_distribution
                        )
                        aligned_profile = compare_transfer_profiles(
                            baseline_test,
                            aligned_distribution,
                            native_intervened,
                            probability_scores=True,
                        )
                        observed = aligned_effect[
                            "mean_absolute_margin_change"
                        ]
                        native_value = native_effect[
                            "mean_absolute_margin_change"
                        ]
                        alignment_null = None
                        observation_effects = None
                        if method == "hungarian_abs":
                            alignment_null = {
                                "null_type": (
                                    "validation_target_circular_shift_refit_"
                                    "hungarian_abs"
                                ),
                                "repeats": null_repeats,
                                "controls": alignment_null_rows,
                                "mean_effect": mean_alignment_null_effect,
                                "mean_profile_to_native": mean_alignment_null_profile,
                                "effects": absolute_alignment_null_effects,
                                "mean": float(
                                    mean_alignment_null_effect[
                                        "mean_absolute_margin_change"
                                    ]
                                ),
                                "std": float(
                                    np.std(absolute_alignment_null_effects)
                                ),
                                "empirical_p": float(
                                    (
                                        1
                                        + sum(
                                            value >= observed
                                            for value in absolute_alignment_null_effects
                                        )
                                    )
                                    / (1 + len(absolute_alignment_null_effects))
                                ),
                            }
                            observation_effects = {
                                "aligned": action_score_observation_effects(
                                    baseline_test,
                                    aligned_distribution,
                                    probability_scores=True,
                                ),
                                "raw": action_score_observation_effects(
                                    baseline_test,
                                    raw_distribution,
                                    probability_scores=True,
                                ),
                                "native": action_score_observation_effects(
                                    baseline_test,
                                    native_intervened,
                                    probability_scores=True,
                                ),
                            }
                        rows.append(
                            {
                                **base,
                                "status": "completed",
                                "alignment_method": method,
                                "alignment_metadata": fitted["metadata"],
                                "source_indices": source_indices,
                                "aligned_target_indices": aligned_indices,
                                "raw_target_indices": source_indices,
                                "target_native_indices": native_indices,
                                "source_circuit_size": len(source_indices),
                                "target_native_size": len(native_indices),
                                "validation_observations": int(len(reference)),
                                "test_observations": int(test_mask.sum()),
                                "mean_absolute_alignment_correlation": alignment[
                                    "mean_absolute_correlation"
                                ],
                                "min_absolute_alignment_correlation": alignment[
                                    "min_absolute_correlation"
                                ],
                                "target_native_effect": native_effect,
                                "aligned_transfer_effect": aligned_effect,
                                "raw_index_effect": raw_effect,
                                "aligned_profile_to_native": aligned_profile,
                                "raw_profile_to_native": raw_profile,
                                "aligned_to_native_ratio": (
                                    float(observed / native_value)
                                    if abs(native_value) > 1e-12
                                    else None
                                ),
                                "random_control": {
                                    "repeats": int(random_controls),
                                    "indices": [
                                        [int(value) for value in indices]
                                        for indices in control_indices
                                    ],
                                    "controls": controls,
                                    "mean_effect": mean_control_effect,
                                    "mean_profile_to_native": mean_control_profile,
                                    "effects": absolute_control_effects,
                                    "mean": float(
                                        mean_control_effect[
                                            "mean_absolute_margin_change"
                                        ]
                                    ),
                                    "std": float(
                                        np.std(absolute_control_effects)
                                    ),
                                    "empirical_p": float(
                                        (
                                            1
                                            + sum(
                                                value >= observed
                                                for value in absolute_control_effects
                                            )
                                        )
                                        / (1 + len(absolute_control_effects))
                                    ),
                                },
                                "alignment_null": alignment_null,
                                "mapping_stability": (
                                    mapping_stability
                                    if method == "hungarian_abs"
                                    else None
                                ),
                                "observation_effects": observation_effects,
                            }
                        )
                except Exception as exc:
                    rows.append(
                        {
                            **base,
                            "status": "failed",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
        failed_rows = sum(row.get("status") == "failed" for row in rows)
        transfer = {
            "schema_version": 3,
            "target_run_id": target_run.run_id,
            "algorithm": target_run.algorithm,
            "market": target_run.market,
            "target_seed": int(target_run.seed),
            "alignment_partition": "validation",
            "evaluation_partition": "test",
            "random_controls": int(random_controls),
            "alignment_null_controls": null_repeats,
            "test_dates": test_dates,
            "alignment_methods": list(methods),
            "rows": rows,
        }
        _write_json(output_dir / "transfer.json", transfer)
        final_status = "completed" if failed_rows == 0 else "failed"
        _write_json(
            status_path,
            {
                "run_id": target_run.run_id,
                "status": final_status,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "failed_rows": failed_rows,
            },
        )
        if failed_rows:
            raise RuntimeError(f"{failed_rows} transfer rows failed")
        return {"status": "completed", "run_dir": str(output_dir), "transfer": transfer}
    except Exception as exc:
        if not status_path.is_file() or _read_json(status_path).get("status") != "failed":
            _write_json(
                status_path,
                {
                    "run_id": target_run.run_id,
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
