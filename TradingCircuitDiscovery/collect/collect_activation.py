from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np

from TradingCircuitDiscovery.collect.dataset import ActivationBank


def collect_activation_bank(
    adapter,
    env,
    output_path: str | Path,
    max_steps: int | None = None,
    concept_fn=None,
    date_fn=None,
    partition_fn=None,
    metadata: dict | None = None,
) -> ActivationBank:
    state = env.reset()
    done = False
    records = []
    steps = 0
    while not done:
        record_t = int(env.t)
        activations = adapter.get_hidden(state)
        action = adapter.get_action(state)
        if concept_fn:
            parameter_count = len(inspect.signature(concept_fn).parameters)
            concepts = concept_fn(env, record_t, action) if parameter_count >= 3 else concept_fn(env, record_t)
        else:
            concepts = {}
        record_date = date_fn(env, record_t) if date_fn else None
        next_state, reward, done = env.step(action)
        record = {
            "t": record_t,
            "state": np.asarray(state, dtype=np.float32),
            "action": int(action),
            "reward": float(reward),
            "concepts": concepts,
            "activations": activations,
        }
        if record_date is not None:
            date_text = record_date.isoformat() if hasattr(record_date, "isoformat") else str(record_date)
            record["date"] = date_text
            if partition_fn:
                record["partition"] = partition_fn(record_date)
        records.append(record)
        state = next_state
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    bank = ActivationBank(records=records, metadata=metadata or {})
    bank.save(output_path)
    return bank
