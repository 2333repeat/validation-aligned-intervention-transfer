from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from TradingCircuitDiscovery.collect.collect_activation import collect_activation_bank
from TradingCircuitDiscovery.collect.dataset import flatten_activation_dataset
from TradingCircuitDiscovery.concepts.market_regime import label_market_regime
from TradingCircuitDiscovery.discovery.circuit_search import CandidateNeuron, greedy_circuit_search
from TradingCircuitDiscovery.discovery.linear_probe import train_temporal_probe
from TradingCircuitDiscovery.discovery.neuron_ranking import rank_neurons_by_activation_difference
from TradingCircuitDiscovery.evaluation.causality_test import compare_action_score_effect, matched_random_circuits
from TradingCircuitDiscovery.evaluation.paired_rollout import run_paired_rollout
from TradingCircuitDiscovery.experiments.manifest import ExperimentRun
from TradingCircuitDiscovery.integrations.policy_value import build_policy_value_adapter
from TradingCircuitDiscovery.integrations.policy_value_env import (
    build_policy_value_environment,
    fit_training_regime_thresholds,
    load_market_data,
    temporal_partition,
)
from TradingCircuitDiscovery.intervention.circuit_edit import CircuitEditor
from TradingCircuitDiscovery.sae.train_sae import evaluate_sae, fit_sae_candidates


DEFAULT_CONFIG = {
    "concepts": ["trend", "high_volatility", "sideway", "long_action", "short_action", "risk_off"],
    "lookback": 20,
    "threshold_quantile": 0.7,
    "top_candidates": 8,
    "max_circuit_size": 4,
    "random_controls": 50,
    "sae_expansion_factors": [2, 4, 8],
    "sae_epochs": 60,
    "sae_patience": 8,
    "sae_l1_lambda": 0.001,
}


def run_directory(output_root: str | Path, run: ExperimentRun) -> Path:
    return Path(output_root) / "runs" / run.algorithm / run.market / f"seed_{run.seed}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _default_layers(algorithm: str) -> list[str]:
    return {
        "PPO": ["actor.fc1"],
        "A2C": ["actor.fc1"],
        "SAC": ["actor.fc1"],
        "DDQN": ["policy_net.fc1", "policy_net.fc2"],
    }[algorithm]


def _algorithm_config(algorithm: str) -> str:
    return f"configs/{algorithm}.yaml"


def _concepts_for_step(env, t: int, action: int, thresholds: dict, lookback: int) -> dict[str, bool]:
    frame = env.account
    regimes = label_market_regime(frame, t=t, thresholds=thresholds, lookback=lookback)
    return {
        **regimes,
        "long_action": bool(action == 1),
        "short_action": bool(action == 2),
        "risk_off": bool(action == 0 and regimes["high_volatility"]),
    }


def _scores(adapter, states, layer: str | None = None, indices=None, values=None, full_values=None) -> np.ndarray:
    states_array = np.asarray(states, dtype=np.float32)
    if getattr(adapter, "supports_batch_scores", False):
        if full_values is not None:
            context = adapter.patch_hidden(layer, torch.as_tensor(full_values, dtype=torch.float32))
        elif indices is not None:
            context = adapter.intervene_hidden(layer, list(indices), values)
        else:
            context = nullcontext()
        with context:
            output = adapter.get_action_scores(states_array).detach().cpu().numpy()
        return output.reshape(1, -1) if output.ndim == 1 else output
    rows = []
    for row_index, state in enumerate(states_array):
        if full_values is not None:
            context = adapter.patch_hidden(layer, torch.as_tensor(full_values[row_index]).reshape(1, -1))
        elif indices is not None:
            replacement = values[row_index] if np.asarray(values).ndim > 1 else values
            context = adapter.intervene_hidden(layer, list(indices), replacement)
        else:
            context = nullcontext()
        with context:
            rows.append(adapter.get_action_scores(state).detach().cpu().numpy())
    return np.vstack(rows)


def _distributions(adapter, states, layer: str | None = None, indices=None, values=None, full_values=None) -> np.ndarray:
    states_array = np.asarray(states, dtype=np.float32)
    if getattr(adapter, "supports_batch_scores", False):
        if full_values is not None:
            context = adapter.patch_hidden(layer, torch.as_tensor(full_values, dtype=torch.float32))
        elif indices is not None:
            context = adapter.intervene_hidden(layer, list(indices), values)
        else:
            context = nullcontext()
        with context:
            output = adapter.get_action_distribution(states_array).detach().cpu().numpy()
        return output.reshape(1, -1) if output.ndim == 1 else output
    rows = []
    for row_index, state in enumerate(states_array):
        if full_values is not None:
            context = adapter.patch_hidden(layer, torch.as_tensor(full_values[row_index]).reshape(1, -1))
        elif indices is not None:
            replacement = values[row_index] if np.asarray(values).ndim > 1 else values
            context = adapter.intervene_hidden(layer, list(indices), replacement)
        else:
            context = nullcontext()
        with context:
            rows.append(adapter.get_action_distribution(state).detach().cpu().numpy())
    return np.vstack(rows)


def _probe_payload(result) -> dict:
    return {
        "coefficients": result.coefficients,
        "intercept": result.intercept,
        "validation": result.validation_metrics,
        "test": result.test_metrics,
        "fit_rows": len(result.fit_indices),
        "validation_rows": len(result.validation_indices),
        "test_rows": len(result.test_indices),
        "feature_mean": result.feature_mean,
        "feature_scale": result.feature_scale,
    }


def _run_layer_concept(adapter, records, layer: str, concept: str, cfg: dict, env_factory) -> dict:
    x, y = flatten_activation_dataset(records, layer=layer, concept=concept)
    partitions = np.asarray([record.get("partition", "excluded") for record in records])
    states = np.vstack([np.asarray(record["state"], dtype=np.float32) for record in records])
    probe = train_temporal_probe(x, y, partitions, random_state=int(cfg["random_state"]))
    discovery_mask = partitions == "discovery"
    validation_mask = partitions == "validation"
    test_mask = partitions == "test"
    ranking = rank_neurons_by_activation_difference(
        x[discovery_mask],
        y[discovery_mask],
        top_k=min(int(cfg["top_candidates"]), x.shape[1]),
    )
    candidates = [CandidateNeuron(layer=layer, index=item.index, score=item.score) for item in ranking]
    discovery_mean = x[discovery_mask].mean(axis=0)
    validation_states = states[validation_mask]
    test_states = states[test_mask]
    baseline_validation = _scores(adapter, validation_states)
    baseline_test = _scores(adapter, test_states)

    def effect_fn(selected):
        indices = [item.index for item in selected]
        intervened = _scores(
            adapter,
            validation_states,
            layer=layer,
            indices=indices,
            values=discovery_mean[indices],
        )
        return compare_action_score_effect(baseline_validation, intervened)["mean_absolute_margin_change"]

    circuit = greedy_circuit_search(candidates, effect_fn=effect_fn, max_size=int(cfg["max_circuit_size"]))
    selected_indices = [item.index for item in circuit.neurons]
    necessity_scores = _scores(
        adapter,
        test_states,
        layer=layer,
        indices=selected_indices,
        values=discovery_mean[selected_indices],
    )
    necessity = compare_action_score_effect(baseline_test, necessity_scores)

    class_means = {
        label: x[discovery_mask & (y == label)].mean(axis=0)
        for label in np.unique(y[discovery_mask])
    }
    test_labels = y[test_mask]
    patch_values = np.vstack(
        [class_means[1 - int(label)][selected_indices] for label in test_labels]
    )
    patched_scores = _scores(
        adapter,
        test_states,
        layer=layer,
        indices=selected_indices,
        values=patch_values,
    )
    patching = compare_action_score_effect(baseline_test, patched_scores)

    random_indices = matched_random_circuits(
        width=x.shape[1],
        size=len(selected_indices),
        repeats=int(cfg["random_controls"]),
        excluded=selected_indices,
        random_state=int(cfg["random_state"]),
    )
    random_effects = []
    for indices in random_indices:
        control_scores = _scores(
            adapter,
            test_states,
            layer=layer,
            indices=indices,
            values=discovery_mean[indices],
        )
        random_effects.append(compare_action_score_effect(baseline_test, control_scores)["mean_absolute_margin_change"])
    targeted_effect = necessity["mean_absolute_margin_change"]
    empirical_p = (1.0 + sum(value >= targeted_effect for value in random_effects)) / (1.0 + len(random_effects))

    economic = run_paired_rollout(
        adapter,
        env_factory=env_factory,
        intervention_fn=lambda current_adapter, _state: current_adapter.intervene_hidden(
            layer,
            selected_indices,
            discovery_mean[selected_indices],
        ),
    )
    return {
        "probe": _probe_payload(probe),
        "ranking": [asdict(item) for item in ranking],
        "selected_indices": selected_indices,
        "selected_neurons": [item.name for item in circuit.neurons],
        "validation_causal_effect": circuit.effect,
        "necessity": necessity,
        "patching": patching,
        "random_control": {
            "effects": random_effects,
            "mean": float(np.mean(random_effects)),
            "std": float(np.std(random_effects)),
            "empirical_p": float(empirical_p),
        },
        "economic": economic,
    }


def _run_sae(adapter, records, layer: str, cfg: dict, concept_labels: dict[str, np.ndarray]) -> dict:
    activations = np.vstack(
        [record["activations"][layer].detach().cpu().numpy().reshape(-1) for record in records]
    ).astype(np.float32)
    partitions = np.asarray([record.get("partition", "excluded") for record in records])
    discovery = torch.tensor(activations[partitions == "discovery"])
    validation = torch.tensor(activations[partitions == "validation"])
    test = torch.tensor(activations[partitions == "test"])
    result = fit_sae_candidates(
        discovery,
        validation,
        expansion_factors=tuple(cfg["sae_expansion_factors"]),
        epochs=int(cfg["sae_epochs"]),
        patience=int(cfg["sae_patience"]),
        l1_lambda=float(cfg["sae_l1_lambda"]),
        random_state=int(cfg["random_state"]),
        device=str(cfg.get("device", "cpu")),
    )
    test_metrics = evaluate_sae(result.model, test)
    feature_effects = {}
    states = np.vstack([np.asarray(record["state"], dtype=np.float32) for record in records])
    test_states = states[partitions == "test"]
    baseline_scores = _scores(adapter, test_states)
    with torch.no_grad():
        _reconstruction, discovery_latent = result.model(discovery)
    for concept, labels in concept_labels.items():
        discovery_labels = labels[partitions == "discovery"]
        if len(np.unique(discovery_labels)) != 2:
            continue
        latent_np = discovery_latent.detach().cpu().numpy()
        feature_scores = np.abs(latent_np[discovery_labels == 1].mean(axis=0) - latent_np[discovery_labels == 0].mean(axis=0))
        feature_index = int(np.argmax(feature_scores))
        edited_hidden = np.vstack(
            [
                CircuitEditor.sae_feature_edit(
                    result.model,
                    test[index : index + 1],
                    feature_indices=[feature_index],
                    scale=0.0,
                ).numpy()
                for index in range(len(test))
            ]
        )
        edited_scores = _scores(adapter, test_states, layer=layer, full_values=edited_hidden)
        feature_effects[concept] = {
            "feature_index": feature_index,
            **compare_action_score_effect(baseline_scores, edited_scores),
        }
    checkpoint_path = cfg.get("sae_checkpoint_path")
    if checkpoint_path:
        torch.save(result.model.state_dict(), checkpoint_path)
    return {
        "latent_dim": result.latent_dim,
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_loss,
        "validation_metrics": result.validation_metrics,
        "test_metrics": test_metrics,
        "candidates": result.candidates,
        "feature_interventions": feature_effects,
    }


def run_checkpoint_experiment(
    run: ExperimentRun,
    project_root: str | Path,
    output_root: str | Path,
    config: dict | None = None,
    adapter=None,
    env_factory=None,
    market_data=None,
) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg["random_state"] = int(run.seed)
    run_dir = run_directory(output_root, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    _write_json(run_dir / "status.json", {"run_id": run.run_id, "status": "running", "started_at": started_at})
    try:
        checkpoint_hash = _sha256(run.checkpoint_path)
        layers = list(getattr(adapter, "layers", []) or _default_layers(run.algorithm))
        if adapter is None:
            adapter = build_policy_value_adapter(
                algorithm=run.algorithm,
                agent_config_path=_algorithm_config(run.algorithm),
                checkpoint_path=run.checkpoint_path,
                layers=layers,
                project_root=project_root,
            )
        if market_data is None:
            market_data = load_market_data(project_root, run.market)
        thresholds = fit_training_regime_thresholds(
            market_data,
            train_end_date="2021-12-31",
            lookback=int(cfg["lookback"]),
            threshold_quantile=float(cfg["threshold_quantile"]),
        )
        if env_factory is None:
            collection_factory = lambda: build_policy_value_environment(
                project_root, run.market, "2022-01-01", "2024-12-31"
            )
            paired_factory = lambda: build_policy_value_environment(
                project_root, run.market, "2024-01-01", "2024-12-31"
            )
        else:
            collection_factory = env_factory
            paired_factory = env_factory
        activation_path = run_dir / "activation" / "oos.pt"
        bank = collect_activation_bank(
            adapter,
            collection_factory(),
            activation_path,
            concept_fn=lambda env, t, action: _concepts_for_step(
                env, t, action, thresholds=thresholds, lookback=int(cfg["lookback"])
            ),
            date_fn=lambda env, t: env.account.loc[t, "Date"],
            partition_fn=temporal_partition,
            metadata={
                "run_id": run.run_id,
                "algorithm": run.algorithm,
                "market": run.market,
                "seed": run.seed,
                "checkpoint_sha256": checkpoint_hash,
                "schema_version": 1,
            },
        )
        probes = {}
        circuits = {}
        interventions = {}
        concept_metrics = {}
        labels_by_concept = {
            concept: np.asarray([int(record["concepts"].get(concept, 0)) for record in bank.records])
            for concept in cfg["concepts"]
        }
        for concept in cfg["concepts"]:
            layer_results = {}
            for layer in layers:
                try:
                    layer_results[layer] = _run_layer_concept(
                        adapter,
                        bank.records,
                        layer,
                        concept,
                        cfg,
                        paired_factory,
                    )
                except ValueError as exc:
                    layer_results[layer] = {"status": "skipped", "reason": str(exc)}
            valid_layers = [
                (layer, payload)
                for layer, payload in layer_results.items()
                if payload.get("status") != "skipped"
            ]
            probes[concept] = {
                layer: payload.get("probe", payload)
                for layer, payload in layer_results.items()
            }
            if not valid_layers:
                concept_metrics[concept] = {"status": "skipped", "layers": layer_results}
                continue
            selected_layer, selected = max(
                valid_layers,
                key=lambda item: item[1]["probe"]["validation"]["balanced_accuracy"],
            )
            circuits[concept] = {
                "layer": selected_layer,
                "indices": selected["selected_indices"],
                "neurons": selected["selected_neurons"],
            }
            interventions[concept] = {
                "necessity": selected["necessity"],
                "patching": selected["patching"],
                "random_control": selected["random_control"],
                "economic": selected["economic"],
            }
            concept_metrics[concept] = {
                "layer": selected_layer,
                "probe": selected["probe"],
                "necessity": selected["necessity"],
                "patching": selected["patching"],
                "random_control": selected["random_control"],
                "economic": selected["economic"],
            }
        sae = {}
        for layer in layers:
            sae_checkpoint = run_dir / "activation" / f"sae_{layer.replace('.', '_')}.pt"
            layer_cfg = {**cfg, "sae_checkpoint_path": sae_checkpoint}
            sae[layer] = _run_sae(adapter, bank.records, layer, layer_cfg, labels_by_concept)
        metrics = {
            "run_id": run.run_id,
            "algorithm": run.algorithm,
            "market": run.market,
            "seed": run.seed,
            "checkpoint_sha256": checkpoint_hash,
            "concepts": concept_metrics,
            "sae": sae,
        }
        _write_json(run_dir / "probes.json", probes)
        _write_json(run_dir / "neuron_circuits.json", circuits)
        _write_json(run_dir / "interventions.json", interventions)
        _write_json(run_dir / "sae.json", sae)
        _write_json(run_dir / "metrics.json", metrics)
        status = {
            "run_id": run.run_id,
            "status": "completed",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "error": None,
        }
        _write_json(run_dir / "status.json", status)
        return {"status": "completed", "run_dir": str(run_dir), "metrics": metrics}
    except Exception as exc:
        _write_json(
            run_dir / "status.json",
            {
                "run_id": run.run_id,
                "status": "failed",
                "started_at": started_at,
                "completed_at": _utc_now(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
