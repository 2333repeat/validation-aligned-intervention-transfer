from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from TradingCircuitDiscovery.collect.dataset import ActivationBank, flatten_activation_dataset
from TradingCircuitDiscovery.evaluation.causality_test import compare_action_score_effect
from TradingCircuitDiscovery.experiments.manifest import ExperimentRun, build_full_manifest
from TradingCircuitDiscovery.experiments.run_checkpoint import (
    _algorithm_config,
    _default_layers,
    _distributions,
    _write_json,
    run_directory,
)
from TradingCircuitDiscovery.integrations.policy_value import build_policy_value_adapter


def recompute_run_distribution_effects(
    run: ExperimentRun,
    project_root: str | Path,
    output_root: str | Path,
    adapter=None,
) -> dict:
    run_dir = run_directory(output_root, run)
    bank = ActivationBank.load(run_dir / "activation" / "oos.pt")
    circuits = json.loads((run_dir / "neuron_circuits.json").read_text(encoding="utf-8"))
    if adapter is None:
        adapter = build_policy_value_adapter(
            algorithm=run.algorithm,
            agent_config_path=_algorithm_config(run.algorithm),
            checkpoint_path=run.checkpoint_path,
            layers=_default_layers(run.algorithm),
            project_root=project_root,
        )
    partitions = np.asarray([record.get("partition", "excluded") for record in bank.records])
    states = np.vstack([np.asarray(record["state"], dtype=np.float32) for record in bank.records])
    discovery_mask = partitions == "discovery"
    test_mask = partitions == "test"
    effects = {}
    for concept, circuit in circuits.items():
        layer = circuit["layer"]
        indices = list(circuit["indices"])
        activations, labels = flatten_activation_dataset(bank.records, layer=layer, concept=concept)
        if len(activations) != len(partitions) or not indices:
            continue
        discovery_mean = activations[discovery_mask].mean(axis=0)
        test_states = states[test_mask]
        baseline = _distributions(adapter, test_states)
        necessity_distribution = _distributions(
            adapter,
            test_states,
            layer=layer,
            indices=indices,
            values=discovery_mean[indices],
        )
        necessity = compare_action_score_effect(
            baseline,
            necessity_distribution,
            probability_scores=True,
        )
        discovery_labels = labels[discovery_mask]
        if len(np.unique(discovery_labels)) != 2:
            continue
        class_means = {
            label: activations[discovery_mask & (labels == label)].mean(axis=0)
            for label in np.unique(discovery_labels)
        }
        patch_values = np.vstack(
            [class_means[1 - int(label)][indices] for label in labels[test_mask]]
        )
        patching_distribution = _distributions(
            adapter,
            test_states,
            layer=layer,
            indices=indices,
            values=patch_values,
        )
        patching = compare_action_score_effect(
            baseline,
            patching_distribution,
            probability_scores=True,
        )
        effects[concept] = {
            "layer": layer,
            "indices": indices,
            "necessity": necessity,
            "patching": patching,
        }
    payload = {
        "run_id": run.run_id,
        "algorithm": run.algorithm,
        "market": run.market,
        "seed": run.seed,
        "distribution_semantics": (
            "native_policy_probability" if run.algorithm in {"PPO", "A2C"}
            else "softmax_normalized_native_score"
        ),
        "concepts": effects,
    }
    _write_json(run_dir / "distribution_effects.json", payload)
    return payload


def recompute_full_distribution_effects(
    runs: list[ExperimentRun],
    project_root: str | Path,
    output_root: str | Path,
) -> dict[str, int]:
    summary = {"expected": len(runs), "completed": 0, "skipped": 0, "failed": 0}
    for run in runs:
        run_dir = run_directory(output_root, run)
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            summary["skipped"] += 1
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            summary["skipped"] += 1
            continue
        try:
            recompute_run_distribution_effects(run, project_root, output_root)
            summary["completed"] += 1
        except Exception:
            summary["failed"] += 1
    return summary


def distribution_summary_succeeded(summary: dict[str, int]) -> bool:
    return (
        int(summary.get("expected", -1)) > 0
        and int(summary.get("completed", -1)) == int(summary.get("expected", -1))
        and int(summary.get("skipped", -1)) == 0
        and int(summary.get("failed", -1)) == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    runs = build_full_manifest(args.project_root, validate_paths=True)
    summary = recompute_full_distribution_effects(runs, args.project_root, args.output_root)
    print(json.dumps(summary))
    return 0 if distribution_summary_succeeded(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
