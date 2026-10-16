from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from TradingCircuitDiscovery.collect.dataset import ActivationBank, flatten_activation_dataset
from TradingCircuitDiscovery.discovery.circuit_search import CandidateNeuron, greedy_circuit_search
from TradingCircuitDiscovery.discovery.linear_probe import train_temporal_probe
from TradingCircuitDiscovery.discovery.neuron_ranking import rank_neurons_by_activation_difference
from TradingCircuitDiscovery.evaluation.attribution import (
    compute_hidden_attributions,
    rank_attribution_baselines,
)
from TradingCircuitDiscovery.evaluation.causality_test import (
    compare_action_score_effect,
    matched_random_circuits,
)
from TradingCircuitDiscovery.evaluation.economic_robustness import (
    block_bootstrap_metric_deltas,
    classify_validation_policy,
)
from TradingCircuitDiscovery.evaluation.intervention_robustness import (
    build_intervention_values,
    select_minimal_near_max,
    standardized_intervention_distance,
)
from TradingCircuitDiscovery.evaluation.paired_rollout import (
    _run_rollout_trace,
    run_paired_rollout_trace,
)
from TradingCircuitDiscovery.evaluation.pnl_test import evaluate_account_returns
from TradingCircuitDiscovery.evaluation.statistics import moving_block_bootstrap_indices
from TradingCircuitDiscovery.experiments.manifest import ExperimentRun
from TradingCircuitDiscovery.experiments.run_checkpoint import (
    _default_layers,
    _distributions,
    _write_json,
    run_directory,
)
from TradingCircuitDiscovery.integrations.policy_value import build_policy_value_adapter
from TradingCircuitDiscovery.integrations.policy_value_env import (
    build_policy_value_environment,
    load_market_data,
)
from TradingCircuitDiscovery.intervention.circuit_edit import CircuitEditor
from TradingCircuitDiscovery.sae.train_sae import evaluate_sae, fit_sae_sparsity_sweep


DEFAULT_SUPPLEMENT_CONFIG = {
    "concepts": ["trend", "high_volatility", "sideway", "long_action", "short_action", "risk_off"],
    "primary_concepts": ["trend", "high_volatility"],
    "action_derived_concepts": ["long_action", "short_action", "risk_off"],
    "top_candidates": 16,
    "circuit_sizes": [2, 4, 8],
    "min_relative_gains": [0.0, 0.01, 0.05],
    "near_max_fraction": 0.95,
    "random_controls": 50,
    "intervention_modes": ["discovery_mean", "zero", "opposite_class", "matched_noise"],
    "integrated_gradient_steps": 8,
    "bootstrap_repeats": 20,
    "bootstrap_block_size": 20,
    "sae_expansion_factors": [2, 4, 8],
    "sae_l1_lambdas": [0.001, 0.01, 0.05],
    "sae_topk_values": [4, 8, 16, 32],
    "sae_epochs": 40,
    "sae_patience": 6,
    "sae_ev_tolerance": 0.02,
    "transaction_costs": [0.0, 0.001, 0.003, 0.005, 0.01],
    "economic_bootstrap_repeats": 1000,
    "economic_block_size": 20,
    "device": "cuda",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _algorithm_config(algorithm: str) -> str:
    return f"configs/{algorithm}.yaml"


def _metric(original, intervened) -> dict[str, float]:
    return compare_action_score_effect(original, intervened, probability_scores=True)


def _jaccard(left, right) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _random_control_payload(
    adapter,
    states,
    layer: str,
    selected_indices: list[int],
    discovery_mean: np.ndarray,
    baseline_distribution: np.ndarray,
    width: int,
    repeats: int,
    random_state: int,
) -> dict:
    controls = matched_random_circuits(
        width=width,
        size=len(selected_indices),
        repeats=repeats,
        excluded=selected_indices,
        random_state=random_state,
    )
    effects = []
    for indices in controls:
        intervened = _distributions(
            adapter,
            states,
            layer=layer,
            indices=indices,
            values=discovery_mean[indices],
        )
        effects.append(_metric(baseline_distribution, intervened)["mean_absolute_margin_change"])
    return {
        "effects": effects,
        "mean": float(np.mean(effects)),
        "std": float(np.std(effects)),
    }


def _buy_hold_metrics(market_data: pd.DataFrame, start_date: str, end_date: str, cost: float) -> dict:
    frame = market_data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    selected = frame.loc[
        (frame["Date"] >= pd.Timestamp(start_date)) & (frame["Date"] <= pd.Timestamp(end_date))
    ].sort_values("Date")
    if len(selected) < 2:
        raise ValueError("buy-and-hold evaluation requires at least two market rows")
    closes = selected["Close"].astype(float).to_numpy()
    returns = closes[1:] / closes[:-1] - 1.0
    metrics = evaluate_account_returns(returns)
    metrics.update(
        {
            "gross_pnl": float(closes[-1] / closes[0] - 1.0),
            "net_pnl": float(closes[-1] * (1.0 - cost) / (closes[0] * (1.0 + cost)) - 1.0),
            "transition_cost": float(cost),
        }
    )
    return metrics


def _concept_family(concept: str, cfg: dict) -> str:
    if concept in cfg.get("primary_concepts", []):
        return "external_primary"
    if concept in cfg.get("action_derived_concepts", []):
        return "action_derived"
    return "derived_or_redundant"


def _probe_logits(values: np.ndarray, probe) -> np.ndarray:
    standardized = (values - probe.feature_mean) / np.maximum(probe.feature_scale, 1e-12)
    return standardized @ probe.coefficients.reshape(-1) + float(probe.intercept)


def _run_sae_layer(
    adapter,
    layer: str,
    activations: np.ndarray,
    partitions: np.ndarray,
    states: np.ndarray,
    labels_by_concept: dict[str, np.ndarray],
    cfg: dict,
    checkpoint_path: Path,
) -> tuple[dict, object]:
    discovery = torch.tensor(activations[partitions == "discovery"], dtype=torch.float32)
    validation = torch.tensor(activations[partitions == "validation"], dtype=torch.float32)
    test = torch.tensor(activations[partitions == "test"], dtype=torch.float32)
    result = fit_sae_sparsity_sweep(
        discovery,
        validation,
        expansion_factors=tuple(cfg["sae_expansion_factors"]),
        l1_lambdas=tuple(cfg["sae_l1_lambdas"]),
        topk_values=tuple(cfg["sae_topk_values"]),
        epochs=int(cfg["sae_epochs"]),
        patience=int(cfg["sae_patience"]),
        random_state=int(cfg["random_state"]),
        device=str(cfg.get("device", "cpu")),
        explained_variance_tolerance=float(cfg.get("sae_ev_tolerance", 0.02)),
    )
    model = result.model.cpu()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    test_metrics = evaluate_sae(model, test)
    with torch.no_grad():
        _reconstruction, discovery_latent = model(discovery)
    latent_values = discovery_latent.numpy()
    baseline_test = _distributions(adapter, states[partitions == "test"])
    feature_rows = {}
    for concept, labels in labels_by_concept.items():
        discovery_labels = labels[partitions == "discovery"]
        if set(np.unique(discovery_labels).tolist()) != {0, 1}:
            continue
        scores = np.abs(
            latent_values[discovery_labels == 1].mean(axis=0)
            - latent_values[discovery_labels == 0].mean(axis=0)
        )
        feature_index = int(np.argmax(scores))
        feature_values = latent_values[:, feature_index]
        auroc = float(roc_auc_score(discovery_labels, feature_values))
        auroc = max(auroc, 1.0 - auroc)
        edited_hidden = CircuitEditor.sae_feature_edit(
            model,
            test,
            feature_indices=[feature_index],
            scale=0.0,
        ).numpy()
        edited_distribution = _distributions(
            adapter,
            states[partitions == "test"],
            layer=layer,
            full_values=edited_hidden,
        )
        feature_rows[concept] = {
            "feature_index": feature_index,
            "concept_auroc": auroc,
            **_metric(baseline_test, edited_distribution),
        }
    return {
        "selected_candidate": result.selected_candidate,
        "candidates": result.candidates,
        "test_metrics": test_metrics,
        "feature_interventions": feature_rows,
        "checkpoint": str(checkpoint_path),
    }, model


def run_supplement_checkpoint(
    run: ExperimentRun,
    project_root: str | Path,
    base_output_root: str | Path,
    output_root: str | Path,
    config: dict | None = None,
    adapter=None,
    market_data: pd.DataFrame | None = None,
    env_factory_for_period=None,
) -> dict:
    cfg = {**DEFAULT_SUPPLEMENT_CONFIG, **(config or {})}
    cfg["random_state"] = int(run.seed)
    run_dir = run_directory(output_root, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    started_at = _utc_now()
    _write_json(status_path, {"run_id": run.run_id, "status": "running", "started_at": started_at})
    try:
        base_run_dir = run_directory(base_output_root, run)
        activation_path = base_run_dir / "activation" / "oos.pt"
        if not activation_path.is_file():
            raise FileNotFoundError(f"Missing base activation bank: {activation_path}")
        bank = ActivationBank.load(activation_path)
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
        if env_factory_for_period is None:
            def env_factory_for_period(start_date, end_date, cost):
                env = build_policy_value_environment(
                    project_root, run.market, start_date, end_date
                )
                env.transitionCost = float(cost)
                return env

        partitions = np.asarray([record.get("partition", "excluded") for record in bank.records])
        states = np.vstack([np.asarray(record["state"], dtype=np.float32) for record in bank.records])
        labels_by_concept = {
            concept: np.asarray(
                [int(record.get("concepts", {}).get(concept, 0)) for record in bank.records],
                dtype=int,
            )
            for concept in cfg["concepts"]
        }
        layer_values = {
            layer: np.vstack(
                [record["activations"][layer].detach().cpu().numpy().reshape(-1) for record in bank.records]
            ).astype(np.float32)
            for layer in layers
        }
        layer_payloads = {}
        layer_models = {}
        gradient_attributions = {}
        for layer, activations in layer_values.items():
            discovery_mask = partitions == "discovery"
            discovery_mean = activations[discovery_mask].mean(axis=0)
            gradient_attributions[layer] = compute_hidden_attributions(
                adapter,
                states[discovery_mask],
                layer,
                discovery_mean,
                ig_steps=int(cfg["integrated_gradient_steps"]),
            )
            sae_path = run_dir / "activation" / f"supplement_sae_{layer.replace('.', '_')}.pt"
            sae_payload, sae_model = _run_sae_layer(
                adapter,
                layer,
                activations,
                partitions,
                states,
                labels_by_concept,
                cfg,
                sae_path,
            )
            layer_payloads[layer] = {
                "width": int(activations.shape[1]),
                "gradient_attributions": gradient_attributions[layer],
                "sae": sae_payload,
            }
            layer_models[layer] = sae_model

        probes = {}
        concept_layers = {}
        discovery_mask = partitions == "discovery"
        validation_mask = partitions == "validation"
        test_mask = partitions == "test"
        for concept in cfg["concepts"]:
            labels = labels_by_concept[concept]
            per_layer = {}
            for layer, activations in layer_values.items():
                try:
                    probe = train_temporal_probe(
                        activations,
                        labels,
                        partitions,
                        random_state=int(cfg["random_state"]),
                    )
                except ValueError as exc:
                    per_layer[layer] = {"status": "skipped", "reason": str(exc)}
                    continue
                probes[(layer, concept)] = probe
                ranking = rank_neurons_by_activation_difference(
                    activations[discovery_mask],
                    labels[discovery_mask],
                    top_k=min(int(cfg["top_candidates"]), activations.shape[1]),
                )
                candidates = [
                    CandidateNeuron(layer=layer, index=item.index, score=item.score)
                    for item in ranking
                ]
                discovery_mean = activations[discovery_mask].mean(axis=0)
                validation_states = states[validation_mask]
                test_states = states[test_mask]
                baseline_validation = _distributions(adapter, validation_states)
                baseline_test = _distributions(adapter, test_states)

                def validation_effect(selected):
                    indices = [item.index for item in selected]
                    values = discovery_mean[indices]
                    intervened = _distributions(
                        adapter,
                        validation_states,
                        layer=layer,
                        indices=indices,
                        values=values,
                    )
                    return _metric(baseline_validation, intervened)["mean_absolute_margin_change"]

                sensitivity = []
                for max_size in cfg["circuit_sizes"]:
                    for min_gain in cfg["min_relative_gains"]:
                        search = greedy_circuit_search(
                            candidates,
                            effect_fn=validation_effect,
                            max_size=min(int(max_size), len(candidates)),
                            min_relative_gain=float(min_gain),
                        )
                        indices = [item.index for item in search.neurons]
                        test_intervened = _distributions(
                            adapter,
                            test_states,
                            layer=layer,
                            indices=indices,
                            values=discovery_mean[indices],
                        )
                        test_effect = _metric(baseline_test, test_intervened)
                        controls = _random_control_payload(
                            adapter,
                            test_states,
                            layer,
                            indices,
                            discovery_mean,
                            baseline_test,
                            width=activations.shape[1],
                            repeats=int(cfg["random_controls"]),
                            random_state=int(cfg["random_state"]) * 1000 + int(max_size) * 10 + int(float(min_gain) * 100),
                        )
                        observed = test_effect["mean_absolute_margin_change"]
                        controls["empirical_p"] = float(
                            (1 + sum(value >= observed for value in controls["effects"]))
                            / (1 + len(controls["effects"]))
                        )
                        sensitivity.append(
                            {
                                "max_size": int(max_size),
                                "min_relative_gain": float(min_gain),
                                "indices": indices,
                                "neurons": [item.name for item in search.neurons],
                                "validation_effect": float(search.effect),
                                "test_effect": test_effect,
                                "random_control": controls,
                            }
                        )
                final = select_minimal_near_max(
                    sensitivity,
                    fraction=float(cfg["near_max_fraction"]),
                )
                selected_indices = list(final["indices"])
                test_labels = labels[test_mask]
                intervention_rows = {}
                for mode_index, mode in enumerate(cfg["intervention_modes"]):
                    replacement = build_intervention_values(
                        mode,
                        activations[discovery_mask],
                        labels[discovery_mask],
                        test_labels,
                        selected_indices,
                        random_state=int(cfg["random_state"]) * 100 + mode_index,
                    )
                    intervened = _distributions(
                        adapter,
                        test_states,
                        layer=layer,
                        indices=selected_indices,
                        values=replacement,
                    )
                    intervention_rows[mode] = {
                        **_metric(baseline_test, intervened),
                        **standardized_intervention_distance(
                            replacement,
                            activations[discovery_mask],
                            selected_indices,
                        ),
                    }
                rankings = rank_attribution_baselines(
                    activations[discovery_mask],
                    labels[discovery_mask],
                    probe.coefficients,
                    gradient_attributions[layer]["gradient_x_activation"],
                    gradient_attributions[layer]["integrated_gradients"],
                    top_k=max(int(value) for value in cfg["circuit_sizes"]),
                )
                attribution_rows = []
                for method, method_ranking in rankings.items():
                    for size in cfg["circuit_sizes"]:
                        indices = method_ranking[: min(int(size), len(method_ranking))]
                        intervened = _distributions(
                            adapter,
                            test_states,
                            layer=layer,
                            indices=indices,
                            values=discovery_mean[indices],
                        )
                        effect = _metric(baseline_test, intervened)
                        controls = _random_control_payload(
                            adapter,
                            test_states,
                            layer,
                            indices,
                            discovery_mean,
                            baseline_test,
                            width=activations.shape[1],
                            repeats=int(cfg["random_controls"]),
                            random_state=int(cfg["random_state"]) * 10000 + len(attribution_rows),
                        )
                        observed = effect["mean_absolute_margin_change"]
                        attribution_rows.append(
                            {
                                "method": method,
                                "requested_size": int(size),
                                "indices": indices,
                                "effect": effect,
                                "random_control_mean": controls["mean"],
                                "empirical_p": float(
                                    (1 + sum(value >= observed for value in controls["effects"]))
                                    / (1 + len(controls["effects"]))
                                ),
                            }
                        )
                bootstrap_rows = []
                final_set = set(selected_indices)
                for repeat in range(int(cfg["bootstrap_repeats"])):
                    discovery_indices = moving_block_bootstrap_indices(
                        int(discovery_mask.sum()),
                        min(int(cfg["bootstrap_block_size"]), int(discovery_mask.sum())),
                        random_state=int(cfg["random_state"]) * 10000 + repeat,
                    )
                    validation_indices = moving_block_bootstrap_indices(
                        int(validation_mask.sum()),
                        min(int(cfg["bootstrap_block_size"]), int(validation_mask.sum())),
                        random_state=int(cfg["random_state"]) * 20000 + repeat,
                    )
                    boot_x = activations[discovery_mask][discovery_indices]
                    boot_y = labels[discovery_mask][discovery_indices]
                    if len(np.unique(boot_y)) != 2:
                        bootstrap_rows.append({"repeat": repeat, "status": "skipped_single_class"})
                        continue
                    boot_ranking = rank_neurons_by_activation_difference(
                        boot_x,
                        boot_y,
                        top_k=min(int(cfg["top_candidates"]), activations.shape[1]),
                    )
                    boot_candidates = [
                        CandidateNeuron(layer=layer, index=item.index, score=item.score)
                        for item in boot_ranking
                    ]
                    boot_mean = boot_x.mean(axis=0)
                    boot_states = validation_states[validation_indices]
                    boot_baseline = baseline_validation[validation_indices]

                    def boot_effect(selected):
                        indices = [item.index for item in selected]
                        intervened = _distributions(
                            adapter,
                            boot_states,
                            layer=layer,
                            indices=indices,
                            values=boot_mean[indices],
                        )
                        return _metric(boot_baseline, intervened)["mean_absolute_margin_change"]

                    boot_search = greedy_circuit_search(
                        boot_candidates,
                        effect_fn=boot_effect,
                        max_size=min(int(final["max_size"]), len(boot_candidates)),
                        min_relative_gain=float(final["min_relative_gain"]),
                    )
                    boot_indices = [item.index for item in boot_search.neurons]
                    bootstrap_rows.append(
                        {
                            "repeat": repeat,
                            "status": "completed",
                            "indices": boot_indices,
                            "jaccard": _jaccard(final_set, boot_indices),
                            "validation_effect": boot_search.effect,
                        }
                    )
                completed_bootstrap = [row for row in bootstrap_rows if row["status"] == "completed"]
                per_layer[layer] = {
                    "status": "completed",
                    "probe": {
                        "validation": probe.validation_metrics,
                        "test": probe.test_metrics,
                        "coefficients": probe.coefficients,
                        "intercept": probe.intercept,
                    },
                    "ranking": [asdict(item) for item in ranking],
                    "circuit_sensitivity": sensitivity,
                    "final_circuit": final,
                    "intervention_robustness": intervention_rows,
                    "attribution_baselines": attribution_rows,
                    "bootstrap_stability": {
                        "rows": bootstrap_rows,
                        "completed": len(completed_bootstrap),
                        "mean_jaccard": float(np.mean([row["jaccard"] for row in completed_bootstrap]))
                        if completed_bootstrap
                        else None,
                    },
                }
            valid = [
                (layer, payload)
                for layer, payload in per_layer.items()
                if payload.get("status") == "completed"
            ]
            if not valid:
                concept_layers[concept] = {
                    "status": "skipped",
                    "family": _concept_family(concept, cfg),
                    "layers": per_layer,
                }
                continue
            selected_layer, selected_payload = max(
                valid,
                key=lambda item: item[1]["probe"]["validation"]["balanced_accuracy"],
            )
            concept_layers[concept] = {
                "status": "completed",
                "family": _concept_family(concept, cfg),
                "selected_layer": selected_layer,
                "layers": per_layer,
                **{
                    key: selected_payload[key]
                    for key in (
                        "circuit_sensitivity",
                        "final_circuit",
                        "intervention_robustness",
                        "attribution_baselines",
                        "bootstrap_stability",
                    )
                },
            }

        validation_metrics, _validation_actions, _validation_returns = _run_rollout_trace(
            adapter,
            env_factory_for_period("2023-01-01", "2023-12-31", 0.003),
        )
        validation_buy_hold = _buy_hold_metrics(
            market_data,
            "2023-01-01",
            "2023-12-31",
            0.003,
        )
        validation_policy = {
            "metrics": validation_metrics,
            "buy_hold": validation_buy_hold,
            "excess_pnl": float(validation_metrics["pnl"] - validation_buy_hold["net_pnl"]),
            "label": classify_validation_policy(validation_metrics, validation_buy_hold),
        }

        for concept, payload in concept_layers.items():
            if payload.get("status") != "completed":
                continue
            layer = payload["selected_layer"]
            selected_indices = list(payload["final_circuit"]["indices"])
            activations = layer_values[layer]
            labels = labels_by_concept[concept]
            discovery_mean = activations[discovery_mask].mean(axis=0)
            edited = activations[test_mask].copy()
            edited[:, selected_indices] = discovery_mean[selected_indices]
            specificity_rows = []
            for evaluation_concept in cfg["concepts"]:
                probe = probes.get((layer, evaluation_concept))
                if probe is None:
                    continue
                original_logits = _probe_logits(activations[test_mask], probe)
                edited_logits = _probe_logits(edited, probe)
                specificity_rows.append(
                    {
                        "evaluation_concept": evaluation_concept,
                        "is_target": evaluation_concept == concept,
                        "mean_absolute_logit_change": float(np.mean(np.abs(edited_logits - original_logits))),
                        "mean_signed_logit_change": float(np.mean(edited_logits - original_logits)),
                    }
                )
            target_values = [row["mean_absolute_logit_change"] for row in specificity_rows if row["is_target"]]
            non_target_values = [row["mean_absolute_logit_change"] for row in specificity_rows if not row["is_target"]]
            payload["concept_specificity"] = {
                "rows": specificity_rows,
                "target_effect": target_values[0] if target_values else None,
                "non_target_mean": float(np.mean(non_target_values)) if non_target_values else None,
                "specificity_gap": (
                    float(target_values[0] - np.mean(non_target_values))
                    if target_values and non_target_values
                    else None
                ),
            }
            economic_rows = []
            for cost in cfg["transaction_costs"]:
                paired = run_paired_rollout_trace(
                    adapter,
                    env_factory=lambda cost=cost: env_factory_for_period(
                        "2024-01-01", "2024-12-31", float(cost)
                    ),
                    intervention_fn=lambda current_adapter, _state, layer=layer, indices=selected_indices, values=discovery_mean[selected_indices]: current_adapter.intervene_hidden(
                        layer, indices, values
                    ),
                )
                bootstrap = block_bootstrap_metric_deltas(
                    paired["baseline_returns"],
                    paired["intervention_returns"],
                    paired["baseline_actions"],
                    paired["intervention_actions"],
                    repeats=int(cfg["economic_bootstrap_repeats"]),
                    block_size=int(cfg["economic_block_size"]),
                    random_state=int(cfg["random_state"]) * 1000 + int(float(cost) * 100000),
                )
                economic_rows.append(
                    {
                        "transition_cost": float(cost),
                        "baseline": paired["baseline"],
                        "intervention": paired["intervention"],
                        "delta": paired["delta"],
                        "block_bootstrap": bootstrap,
                    }
                )
            payload["economic_robustness"] = economic_rows

        supplement = {
            "schema_version": 1,
            "run_id": run.run_id,
            "algorithm": run.algorithm,
            "market": run.market,
            "seed": run.seed,
            "base_activation": str(activation_path),
            "protocol": cfg,
            "validation_policy": validation_policy,
            "layers": layer_payloads,
            "concepts": concept_layers,
        }
        _write_json(run_dir / "supplement.json", supplement)
        _write_json(
            status_path,
            {
                "run_id": run.run_id,
                "status": "completed",
                "started_at": started_at,
                "completed_at": _utc_now(),
                "error": None,
            },
        )
        return {"status": "completed", "run_dir": str(run_dir), "supplement": supplement}
    except Exception as exc:
        _write_json(
            status_path,
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

