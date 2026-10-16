from __future__ import annotations

from contextlib import nullcontext

from TradingCircuitDiscovery.evaluation.pnl_test import evaluate_account_frame


def _run_rollout_trace(adapter, env, intervention_fn=None, max_steps: int | None = None) -> tuple[dict, list[int], list[float]]:
    state = env.reset()
    done = False
    actions: list[int] = []
    steps = 0
    while not done:
        context = intervention_fn(adapter, state) if intervention_fn else nullcontext()
        with context:
            action = int(adapter.get_action(state))
        state, _reward, done = env.step(action)
        actions.append(action)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    metrics = evaluate_account_frame(env.account)
    returns = env.account["Returns"].to_numpy(dtype=float)[: len(actions)].tolist()
    return metrics, actions, returns


def _run_rollout(adapter, env, intervention_fn=None, max_steps: int | None = None) -> tuple[dict, list[int]]:
    metrics, actions, _returns = _run_rollout_trace(
        adapter, env, intervention_fn=intervention_fn, max_steps=max_steps
    )
    return metrics, actions


def run_paired_rollout(adapter, env_factory, intervention_fn, max_steps: int | None = None) -> dict:
    baseline_metrics, baseline_actions = _run_rollout(adapter, env_factory(), max_steps=max_steps)
    intervention_metrics, intervention_actions = _run_rollout(
        adapter,
        env_factory(),
        intervention_fn=intervention_fn,
        max_steps=max_steps,
    )
    common_metrics = sorted(set(baseline_metrics) & set(intervention_metrics))
    delta = {
        key: float(intervention_metrics[key] - baseline_metrics[key])
        for key in common_metrics
        if isinstance(baseline_metrics[key], (int, float))
    }
    return {
        "baseline": baseline_metrics,
        "intervention": intervention_metrics,
        "delta": delta,
        "baseline_actions": baseline_actions,
        "intervention_actions": intervention_actions,
    }


def run_paired_rollout_trace(adapter, env_factory, intervention_fn, max_steps: int | None = None) -> dict:
    baseline_metrics, baseline_actions, baseline_returns = _run_rollout_trace(
        adapter, env_factory(), max_steps=max_steps
    )
    intervention_metrics, intervention_actions, intervention_returns = _run_rollout_trace(
        adapter,
        env_factory(),
        intervention_fn=intervention_fn,
        max_steps=max_steps,
    )
    common_metrics = sorted(set(baseline_metrics) & set(intervention_metrics))
    delta = {
        key: float(intervention_metrics[key] - baseline_metrics[key])
        for key in common_metrics
        if isinstance(baseline_metrics[key], (int, float))
    }
    return {
        "baseline": baseline_metrics,
        "intervention": intervention_metrics,
        "delta": delta,
        "baseline_returns": baseline_returns,
        "intervention_returns": intervention_returns,
        "baseline_actions": baseline_actions,
        "intervention_actions": intervention_actions,
    }
