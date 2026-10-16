from __future__ import annotations

import numpy as np

from TradingCircuitDiscovery.evaluation.pnl_test import evaluate_account_returns
from TradingCircuitDiscovery.evaluation.statistics import moving_block_bootstrap_indices


def classify_validation_policy(policy_metrics: dict, buy_hold_metrics: dict) -> str:
    excess_pnl = float(policy_metrics["pnl"]) - float(buy_hold_metrics["net_pnl"])
    return "success" if excess_pnl > 0 and float(policy_metrics["sharpe"]) > 0 else "underperforming"


def block_bootstrap_metric_deltas(
    baseline_returns,
    intervention_returns,
    baseline_actions,
    intervention_actions,
    repeats: int = 1000,
    block_size: int = 20,
    confidence: float = 0.95,
    random_state: int = 1,
) -> dict[str, dict[str, float]]:
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    intervention = np.asarray(intervention_returns, dtype=np.float64)
    baseline_actions = np.asarray(baseline_actions, dtype=int)
    intervention_actions = np.asarray(intervention_actions, dtype=int)
    if baseline.shape != intervention.shape or baseline.ndim != 1 or len(baseline) == 0:
        raise ValueError("paired return traces must be non-empty one-dimensional arrays")
    if baseline_actions.shape != baseline.shape or intervention_actions.shape != intervention.shape:
        raise ValueError("action traces must align with returns")
    metrics = ("pnl", "sharpe", "mdd", "turnover")
    samples = {metric: [] for metric in metrics}
    for repeat in range(int(repeats)):
        indices = moving_block_bootstrap_indices(
            len(baseline),
            block_size=min(int(block_size), len(baseline)),
            random_state=int(random_state) * 100000 + repeat,
        )
        baseline_metric = evaluate_account_returns(baseline[indices], baseline_actions[indices])
        intervention_metric = evaluate_account_returns(
            intervention[indices], intervention_actions[indices]
        )
        for metric in metrics:
            samples[metric].append(intervention_metric[metric] - baseline_metric[metric])
    alpha = (1.0 - confidence) / 2.0
    return {
        metric: {
            "mean": float(np.mean(values)),
            "ci_low": float(np.quantile(values, alpha)),
            "ci_high": float(np.quantile(values, 1.0 - alpha)),
        }
        for metric, values in samples.items()
    }

