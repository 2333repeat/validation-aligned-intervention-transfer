from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from TradingCircuitDiscovery.collect.dataset import ActivationBank, flatten_activation_dataset
from TradingCircuitDiscovery.discovery.circuit_search import CandidateNeuron, greedy_circuit_search
from TradingCircuitDiscovery.discovery.graph_builder import build_circuit_graph
from TradingCircuitDiscovery.discovery.linear_probe import train_linear_probe
from TradingCircuitDiscovery.discovery.neuron_ranking import rank_neurons_by_activation_difference
from TradingCircuitDiscovery.evaluation.circuit_metric import (
    circuit_causal_score,
    circuit_sparsity,
    circuit_stability,
)
from TradingCircuitDiscovery.evaluation.pnl_test import evaluate_account_returns
from TradingCircuitDiscovery.evaluation.report import generate_report
from TradingCircuitDiscovery.experiments.manifest import ExperimentRun
from TradingCircuitDiscovery.experiments.run_checkpoint import run_checkpoint_experiment
from TradingCircuitDiscovery.models.sae import SparseAutoencoder
from TradingCircuitDiscovery.sae.feature_extract import extract_top_features
from TradingCircuitDiscovery.utils.config import load_config
from TradingCircuitDiscovery.visualization.circuit_graph import draw_circuit_graph
from TradingCircuitDiscovery.visualization.heatmap import draw_heatmap


def _synthetic_records(layer: str = "actor.fc1") -> list[dict]:
    records = []
    for i in range(12):
        concept = int(i >= 6)
        activation = torch.tensor([[float(concept * 2 + i * 0.01), float(1 - concept), float(i % 3)]])
        records.append(
            {
                "date": f"2022-01-{i + 1:02d}",
                "state": np.array([i, i + 1], dtype=np.float32),
                "action": int(i % 3),
                "reward": float((i % 4) - 1) / 100.0,
                "concepts": {"trend": concept, "high_volatility": int(i % 2 == 0)},
                "activations": {layer: activation},
            }
        )
    return records


def run_pipeline(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    project_root: str | Path | None = None,
    algorithm: str | None = None,
    market: str | None = None,
    seed: int = 1,
) -> dict:
    cfg = load_config(config_path)
    configured_market = cfg.get("dataset", {}).get("market", "UNKNOWN")
    out = Path(output_dir or cfg.get("output", {}).get("dir", "TradingCircuitDiscovery/outputs/default"))
    out.mkdir(parents=True, exist_ok=True)
    layer = cfg.get("activation", {}).get("layers", ["actor.fc1"])[0]

    if not dry_run:
        if project_root is None or algorithm is None or market is None:
            raise ValueError("Non-dry-run execution requires project_root, algorithm and market")
        root = Path(project_root).expanduser().resolve()
        normalized_algorithm = algorithm.upper()
        run_id = f"{normalized_algorithm}-{market}-cuda-s{int(seed)}"
        agent_dir = root / "result" / run_id / "agent"
        run = ExperimentRun(
            run_id=run_id,
            algorithm=normalized_algorithm,
            market=market,
            seed=int(seed),
            checkpoint_path=agent_dir / "model" / "best_model.pth",
            account_path=agent_dir / "agent_testAccount.csv",
        )
        if not run.checkpoint_path.is_file() or not run.account_path.is_file():
            raise FileNotFoundError(f"Canonical run files not found for {run_id}")
        return run_checkpoint_experiment(run, project_root=root, output_root=out, config=cfg)

    records = _synthetic_records(layer=layer)
    bank_path = out / "activation_bank" / configured_market / "dry_run.pt"
    ActivationBank(records, metadata={"market": configured_market, "dry_run": True}).save(bank_path)

    x, y = flatten_activation_dataset(records, layer=layer, concept="trend")
    probe = train_linear_probe(x, y)
    ranked = rank_neurons_by_activation_difference(x, y, top_k=3)
    candidates = [CandidateNeuron(layer=layer, index=item.index, score=item.score) for item in ranked]
    circuit = greedy_circuit_search(candidates, effect_fn=lambda selected: sum(item.score for item in selected), max_size=2)
    circuit_neurons = [item.name for item in circuit.neurons]
    circuits = {"trend_circuit": circuit_neurons}

    sae = SparseAutoencoder(input_dim=x.shape[1], latent_dim=min(6, max(2, x.shape[1] * 2)))
    features = extract_top_features(sae, torch.tensor(x, dtype=torch.float32), top_k=2)
    returns = np.asarray([record["reward"] for record in records])
    actions = np.asarray([record["action"] for record in records])
    pnl_stats = evaluate_account_returns(returns, actions=actions)
    metrics = {
        **pnl_stats,
        "probe_accuracy": probe.accuracy,
        "causal_score": circuit_causal_score([0.8, 0.2], [0.4, 0.6], action_index=0),
        "sparsity": circuit_sparsity(circuit_neurons),
        "stability": circuit_stability(circuit_neurons, circuit_neurons),
        "sae_top_features": [int(i) for i in features["indices"].tolist()],
    }

    graph = build_circuit_graph("trend", "trend_circuit", circuit_neurons, "long_probability_change")
    generate_report(metrics=metrics, circuits=circuits, output_dir=out)
    draw_circuit_graph(graph, out / "circuit_graph.png")
    draw_heatmap(x, out / "heatmap.png")
    return {"status": "ok", "output_dir": str(out), "activation_bank": str(bank_path)}


def run_multi_market(config_paths: list[str | Path], output_dir: str | Path, dry_run: bool = False) -> dict:
    base = Path(output_dir)
    markets: list[str] = []
    runs = []
    for config_path in config_paths:
        cfg = load_config(config_path)
        market = cfg.get("dataset", {}).get("market", Path(config_path).stem)
        markets.append(market)
        runs.append(
            run_pipeline(
                config_path=config_path,
                output_dir=base / market,
                dry_run=dry_run,
            )
        )
    return {"status": "ok", "markets": markets, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--multi-market-configs", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--market", default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.multi_market_configs:
        result = run_multi_market(args.multi_market_configs, output_dir=args.output_dir or "TradingCircuitDiscovery/outputs/multi", dry_run=args.dry_run)
    else:
        result = run_pipeline(
            args.config,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            project_root=args.project_root,
            algorithm=args.algorithm,
            market=args.market,
            seed=args.seed,
        )
    print(result)


if __name__ == "__main__":
    main()
